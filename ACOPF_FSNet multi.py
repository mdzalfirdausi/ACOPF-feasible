#!/usr/bin/env python3
"""
ACOPF FSNet (Feasibility-Seeking Neural Network) Training Script
Optimized for CUDA Acceleration / Intel i7 Hybrid Architecture
"""
import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# --- MODEL DEFINITION ---
class BranchedBaselineQCQPMLP(nn.Module):
    """
    Version 1: Separated Neural Networks for different variables.
    Input:
        Pd: [B, nbus]
        Qd: [B, nbus]
    Output:
        v:  [B, 2*nbus] (Rectangular voltages)
        pg: [B, ngen]   (Active generation)
        qg: [B, ngen]   (Reactive generation)
    """
    def __init__(self, nbus: int, ngen: int, slack_imag_idx: int, hidden: int = 256):
        super().__init__()
        self.nbus = nbus
        self.ngen = ngen
        self.in_dim = 2 * nbus
        self.slack_imag_idx = int(slack_imag_idx)

        # ----------------------------------------------------
        # Branch 1: Voltage Variables NN 
        # ----------------------------------------------------
        self.net_v = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * nbus),
        )

        # ----------------------------------------------------
        # Branch 2: Active Power Variables NN
        # ----------------------------------------------------
        self.net_pg = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ngen),
        )

        # ----------------------------------------------------
        # Branch 3: Reactive Power Variables NN
        # ----------------------------------------------------
        self.net_qg = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ngen),
        )

    def forward(self, Pd: torch.Tensor, Qd: torch.Tensor, problem: dict) -> tuple:
        B = Pd.shape[0]
        x = torch.cat([Pd, Qd], dim=-1)

        # 1. Independent Predictions
        v_raw = self.net_v(x)
        pg_raw = self.net_pg(x)
        qg_raw = self.net_qg(x)

        # 2. Bound Voltages to [-Vmax, Vmax] using Tanh for smooth gradients
        Vmax_b = problem["Vmax"].reshape(1, -1).expand(B, -1)
        Vmax_full = torch.cat([Vmax_b, Vmax_b], dim=-1) # For real and imaginary parts
        v = torch.tanh(v_raw) * Vmax_full

        # Constraint (2m): Enforce slack imaginary part = 0 exactly
        v_clone = v.clone()
        v_clone[:, self.slack_imag_idx] = 0.0
        v = v_clone

        # 3. Bound Generation strictly between [min, max] using Sigmoid
        pmax_b = problem["pmax"].reshape(1, -1).expand(B, -1)
        pmin_b = problem["pmin"].reshape(1, -1).expand(B, -1)
        qmax_b = problem["qmax"].reshape(1, -1).expand(B, -1)
        qmin_b = problem["qmin"].reshape(1, -1).expand(B, -1)

        pg = pmin_b + torch.sigmoid(pg_raw) * (pmax_b - pmin_b)
        qg = qmin_b + torch.sigmoid(qg_raw) * (qmax_b - qmin_b)

        return v, pg, qg
    
# --- UTILS & LOSS FUNCTIONS ---
def batch_Mv(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.einsum('kij,bj->bki', M, v)

def quad_batch_stack(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,kij,bj->bk", v, M, v)

def compute_fsnet_qcqp_smax_loss(model, Pd_batch, Qd_batch, problem, weights, seek_steps=5, seek_lr=1e-3):
    B = Pd_batch.shape[0]
    
    # --------------------------------------------------------
    # 1. FORWARD PASS (Initial Guess y_0)
    # --------------------------------------------------------
    v_0, pg_0, qg_0 = model(Pd_batch, Qd_batch, problem)

    # Unpack Problem Matrices
    M_p, M_q = problem["M_p"], problem["M_q"]
    M_pf, M_qf = problem["M_pf"], problem["M_qf"]
    M_pt, M_qt = problem["M_pt"], problem["M_qt"]
    M_c, M_s, M_v = problem["M_c"], problem["M_s"], problem["M_v"]
    C_g = problem["C_g"]
    
    smax = problem["smax"].unsqueeze(0).expand(B, -1)
    angmax = problem["angmax"].unsqueeze(0).expand(B, -1)
    angmin = problem["angmin"].unsqueeze(0).expand(B, -1)
    Vmin = problem["Vmin"].unsqueeze(0).expand(B, -1)
    Vmax = problem["Vmax"].unsqueeze(0).expand(B, -1)
    pmax = problem["pmax"].unsqueeze(0).expand(B, -1)
    pmin = problem["pmin"].unsqueeze(0).expand(B, -1)
    qmax = problem["qmax"].unsqueeze(0).expand(B, -1)
    qmin = problem["qmin"].unsqueeze(0).expand(B, -1)
    c2 = problem["c2"].unsqueeze(0).expand(B, -1)
    c1 = problem["c1"].unsqueeze(0).expand(B, -1)
    c0 = problem["c0"].unsqueeze(0).expand(B, -1) if "c0" in problem else 0.0

    # --------------------------------------------------------
    # 2. FSNET FEASIBILITY SEEKING (Differentiable Inner Loop)
    # --------------------------------------------------------
    is_training = torch.is_grad_enabled()

    if not is_training:
        # During validation, detach from the frozen network and enable gradients locally
        v = v_0.detach().requires_grad_(True)
        pg = pg_0.detach().requires_grad_(True)
        qg = qg_0.detach().requires_grad_(True)
    else:
        # During training, keep the variables connected to the network's computation graph
        v = v_0
        pg = pg_0
        qg = qg_0
    
    with torch.enable_grad(): # Force autograd to be active for the seeking loop
        for _ in range(seek_steps):
            # A. Evaluate Constraints on Current State
            vp = quad_batch_stack(v, M_p); vq = quad_batch_stack(v, M_q)
            pf = quad_batch_stack(v, M_pf); qf = quad_batch_stack(v, M_qf)
            pt = quad_batch_stack(v, M_pt); qt = quad_batch_stack(v, M_qt)
            vc = quad_batch_stack(v, M_c); vs = quad_batch_stack(v, M_s)
            vv = quad_batch_stack(v, M_v)
            
            h_p = (pg @ C_g.T) - Pd_batch - vp
            h_q = (qg @ C_g.T) - Qd_batch - vq
            g_sf = (pf**2 + qf**2) - smax**2
            g_st = (pt**2 + qt**2) - smax**2
            g_ang_min = torch.tan(angmin) * vc - vs
            g_ang_max = vs - torch.tan(angmax) * vc
            g_v_max = vv - (Vmax**2)
            g_v_min = (Vmin**2) - vv
            g_pg_max = pg - pmax; g_pg_min = pmin - pg
            g_qg_max = qg - qmax; g_qg_min = qmin - qg
            
            # Sum up all violations to create the Feasibility Objective P(y)
            viol_loss = (
                h_p.pow(2).mean() + h_q.pow(2).mean() +
                F.relu(g_sf).pow(2).mean() + F.relu(g_st).pow(2).mean() +
                F.relu(g_ang_min).pow(2).mean() + F.relu(g_ang_max).pow(2).mean() +
                F.relu(g_v_max).pow(2).mean() + F.relu(g_v_min).pow(2).mean() +
                F.relu(g_pg_max).pow(2).mean() + F.relu(g_pg_min).pow(2).mean() +
                F.relu(g_qg_max).pow(2).mean() + F.relu(g_qg_min).pow(2).mean()
            )
            
            # B. Compute Differentiable Gradients
            # create_graph=is_training allows backprop to the NN during training, but saves memory in val
            grad_v, grad_pg, grad_qg = torch.autograd.grad(
                viol_loss, (v, pg, qg), create_graph=is_training, retain_graph=True
            )
            
            # C. Take a gradient descent step (moving closer to feasibility)
            v = v - seek_lr * grad_v
            pg = pg - seek_lr * grad_pg
            qg = qg - seek_lr * grad_qg

    # --------------------------------------------------------
    # 3. FINAL TASK LOSS EVALUATION ON \hat{y} (Post-Seeking)
    # --------------------------------------------------------
    vp_f = quad_batch_stack(v, M_p); vq_f = quad_batch_stack(v, M_q)
    pf_f = quad_batch_stack(v, M_pf); qf_f = quad_batch_stack(v, M_qf)
    pt_f = quad_batch_stack(v, M_pt); qt_f = quad_batch_stack(v, M_qt)
    vc_f = quad_batch_stack(v, M_c); vs_f = quad_batch_stack(v, M_s)
    vv_f = quad_batch_stack(v, M_v)

    h_p_f = (pg @ C_g.T) - Pd_batch - vp_f
    h_q_f = (qg @ C_g.T) - Qd_batch - vq_f
    g_sf_f = (pf_f**2 + qf_f**2) - smax**2
    g_st_f = (pt_f**2 + qt_f**2) - smax**2
    g_ang_min_f = torch.tan(angmin) * vc_f - vs_f
    g_ang_max_f = vs_f - torch.tan(angmax) * vc_f
    g_v_max_f = vv_f - (Vmax**2)
    g_v_min_f = (Vmin**2) - vv_f
    g_pg_max_f = pg - pmax; g_pg_min_f = pmin - pg
    g_qg_max_f = qg - qmax; g_qg_min_f = qmin - qg

    # Objective Cost at the feasible point
    cost_per_gen = c2 * (pg ** 2) + c1 * pg + c0
    obj = cost_per_gen.sum(dim=1).mean()

    # --- UPDATED TO MATCH BASELINE WEIGHT STRUCTURE ---
    loss_eq_p = h_p_f.pow(2).mean()
    loss_eq_q = h_q_f.pow(2).mean()

    loss_ineq = (
        F.relu(g_sf_f).pow(2).mean() + F.relu(g_st_f).pow(2).mean() +
        F.relu(g_ang_min_f).pow(2).mean() + F.relu(g_ang_max_f).pow(2).mean() +
        F.relu(g_v_max_f).pow(2).mean() + F.relu(g_v_min_f).pow(2).mean() +
        F.relu(g_pg_max_f).pow(2).mean() + F.relu(g_pg_min_f).pow(2).mean() +
        F.relu(g_qg_max_f).pow(2).mean() + F.relu(g_qg_min_f).pow(2).mean()
    )

    total_loss = (
        (weights["primal_eq_p"] * loss_eq_p) + 
        (weights["primal_eq_q"] * loss_eq_q) + 
        (weights["primal_ineq"] * loss_ineq) + 
        (weights["obj"] * obj)
    )

    # --------------------------------------------------------
    # DIAGNOSTICS FOR BENCHMARKING
    # --------------------------------------------------------
    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "loss_primal": (loss_eq_p + loss_eq_q + loss_ineq).detach().item(),
        "obj_cost": obj.detach().item(),
        
        "max_h_p": h_p_f.abs().max().detach().item(),
        "max_h_q": h_q_f.abs().max().detach().item(),
        "max_thermal": torch.max(F.relu(g_sf_f).max(), F.relu(g_st_f).max()).detach().item(),
        "max_v_viol": torch.max(F.relu(g_v_max_f).max(), F.relu(g_v_min_f).max()).detach().item(),
        "max_gen_viol": torch.max(
            torch.max(F.relu(g_pg_max_f).max(), F.relu(g_pg_min_f).max()),
            torch.max(F.relu(g_qg_max_f).max(), F.relu(g_qg_min_f).max())
        ).detach().item()
    }

    return total_loss, diagnostics

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    # 0. Hardware Device Discovery & Optimization
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA Hardware Acceleration Active: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        max_threads = os.cpu_count() or 1
        torch.set_num_threads(max_threads)
        print(f"Running on CPU Profile. Thread threshold dynamically established at {max_threads} loops.")

    # 1. Load Data
    case_name = 'pglib_opf_case14_ieee'
    total_samples = 10000
    dataset_path = f'./dataset/{case_name}_{total_samples}.pt'
    
    try:
        problem = torch.load(dataset_path, map_location=device)
    except FileNotFoundError:
        print(f"CRITICAL: Admittance topology dataset not found at target: {dataset_path}")
        sys.exit(1)

    # 2. Extract Data Split Slices
    actual_total_samples = problem["Pd_all"].shape[0] 
    train_size = int(0.8 * actual_total_samples)
    val_size = int(0.1 * actual_total_samples)

    print(f"Problem Geometry Linked -> Matrix Samples: {actual_total_samples}")
    
    # Slice arrays and ensure deployment to the designated target device
    train_Pd = problem["Pd_all"][:train_size].to(device)
    train_Qd = problem["Qd_all"][:train_size].to(device)
    # --- Slice VAL arrays and deploy to the target device ---
    val_Pd = problem["Pd_all"][train_size:train_size + val_size].to(device)
    val_Qd = problem["Qd_all"][train_size:train_size + val_size].to(device)

    # Transition background system tensors to matching target device
    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            problem[key] = value.to(device)

    # 3. Setup Dataset Pipeline
    batch_size = 1024 
    train_dataset = TensorDataset(train_Pd, train_Qd)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Model Instantiation & Parameter Configurations
    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()

    model_fsnet = BranchedBaselineQCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        slack_imag_idx=slack_imag_idx
    ).to(device)

    optimizer_fsnet = optim.Adam(model_fsnet.parameters(), lr=1e-3)

    # --- UPDATED FSNET LOSS WEIGHTS ---
    loss_weights_fsnet = {
        "primal_eq_p": 1000.0,   # Matches baseline "eq_p"
        "primal_eq_q": 1000.0,   # Matches baseline "eq_q"
        "primal_ineq": 1.0,      # Matches baseline inequalities
        "obj": 0.01              # Generation cost weight matching baseline
    }

    epochs = 10000
    # --- Initialize checkpoint trackers ---
    best_val_loss = float('inf')
    model_save_path = f"./model/best_fsnet_model_{case_name}_{epochs}epochs.pth"

    # 5. Optimization Loop Execution
    print("\nBeginning execution of parallelized training matrix loops for FSNet...")
    start_time = time.time()
    for epoch in range(epochs):
        model_fsnet.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer_fsnet.zero_grad()
            
            # Run FSNet with 5 unrolled seeking steps
            loss, diag = compute_fsnet_qcqp_smax_loss(
                model=model_fsnet, 
                Pd_batch=Pd_batch, 
                Qd_batch=Qd_batch, 
                problem=problem, 
                weights=loss_weights_fsnet,
                seek_steps=5,     
                seek_lr=1e-2      
            )
            
            loss.backward()
            
            # Clipping is mandatory because second-order autograd gradients can explode
            torch.nn.utils.clip_grad_norm_(model_fsnet.parameters(), 10.0)
            optimizer_fsnet.step()
            
        if epoch % 10 == 0:  
            # 1. Switch to evaluation mode and freeze gradients
            model_fsnet.eval()
            with torch.no_grad():
                # Evaluate the entire validation set at once
                val_loss, val_diag = compute_fsnet_qcqp_smax_loss(model_fsnet, val_Pd, val_Qd, problem, loss_weights_fsnet)

            # 2. Checkpointing Logic: If this is the lowest validation loss we've seen, save it!
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_fsnet.state_dict(), model_save_path)
                saved_flag = " [*SAVED BEST*]"
            else:
                saved_flag = ""

            print(f"Epoch {epoch:4d} | Val Loss: {val_loss:.4f} | Val Cost: {val_diag['obj_cost']:7.2f} | "
                  f"Val Max P-Miss: {val_diag['max_h_p']:.4f} | Val Max Q-Miss: {val_diag['max_h_q']:.4f} | "
                  f"Val Max Gen Viol: {val_diag['max_gen_viol']:.4f} | Val Max Thermal: {val_diag['max_thermal']:.4f}{saved_flag}")
    
    end_time = time.time()
    total_time_seconds = end_time - start_time
    # Format into Hours, Minutes, and Seconds 
    hours, remainder = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "="*50)
    print(f"Training Complete!")
    print(f"Total Training Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"Best model weights saved to: {model_save_path}")
    print("="*50 + "\n")
