#!/usr/bin/env python3
"""
ACOPF DC3 (Deep Constraint Completion & Correction) Training Script
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

def compute_dc3_qcqp_smax_loss(model, Pd_batch, Qd_batch, problem, weights, corr_steps=10, corr_lr=1e-3):
    B = Pd_batch.shape[0]
    
    # --------------------------------------------------------
    # 1. FORWARD PASS (Network Prediction)
    # --------------------------------------------------------
    v_pred, pg_pred, qg_pred = model(Pd_batch, Qd_batch, problem)

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
    c2 = problem["c2"].unsqueeze(0).expand(B, -1)
    c1 = problem["c1"].unsqueeze(0).expand(B, -1)
    c0 = problem["c0"].unsqueeze(0).expand(B, -1) if "c0" in problem else 0.0

    # --------------------------------------------------------
    # 2. DC3 CORRECTION PHASE (Inner Optimization Loop)
    # --------------------------------------------------------
    v_c = v_pred.detach().clone().requires_grad_(True)
    pg_c = pg_pred.detach().clone().requires_grad_(True)
    qg_c = qg_pred.detach().clone().requires_grad_(True)
    
    # Define an inner optimizer solely for the correction steps
    optimizer_corr = torch.optim.Adam([v_c, pg_c, qg_c], lr=corr_lr)
    
    with torch.enable_grad():
        for _ in range(corr_steps):
            optimizer_corr.zero_grad()
            
            # Evaluate Physics on the *Correction* variables
            vp_c = quad_batch_stack(v_c, M_p)
            vq_c = quad_batch_stack(v_c, M_q)
            pf_c = quad_batch_stack(v_c, M_pf); qf_c = quad_batch_stack(v_c, M_qf)
            pt_c = quad_batch_stack(v_c, M_pt); qt_c = quad_batch_stack(v_c, M_qt)
            vc_c = quad_batch_stack(v_c, M_c); vs_c = quad_batch_stack(v_c, M_s)
            vv_c = quad_batch_stack(v_c, M_v)
            
            # Constraints
            h_p_c = (pg_c @ C_g.T) - Pd_batch - vp_c
            h_q_c = (qg_c @ C_g.T) - Qd_batch - vq_c
            g_sf_c = (pf_c**2 + qf_c**2) - smax**2
            g_st_c = (pt_c**2 + qt_c**2) - smax**2
            g_ang_min_c = torch.tan(angmin) * vc_c - vs_c
            g_ang_max_c = vs_c - torch.tan(angmax) * vc_c
            g_v_max_c = vv_c - (Vmax**2)
            g_v_min_c = (Vmin**2) - vv_c
            
            # Sum up all violations to create a repair gradient
            viol_loss = (
                h_p_c.pow(2).mean() + h_q_c.pow(2).mean() +
                F.relu(g_sf_c).pow(2).mean() + F.relu(g_st_c).pow(2).mean() +
                F.relu(g_ang_min_c).pow(2).mean() + F.relu(g_ang_max_c).pow(2).mean() +
                F.relu(g_v_max_c).pow(2).mean() + F.relu(g_v_min_c).pow(2).mean()
            )
            
            viol_loss.backward()
            optimizer_corr.step()

    # --------------------------------------------------------
    # 3. STANDARD PRIMAL EVALUATION (On original NN output)
    # --------------------------------------------------------
    vp = quad_batch_stack(v_pred, M_p); vq = quad_batch_stack(v_pred, M_q)
    pf = quad_batch_stack(v_pred, M_pf); qf = quad_batch_stack(v_pred, M_qf)
    pt = quad_batch_stack(v_pred, M_pt); qt = quad_batch_stack(v_pred, M_qt)
    vc = quad_batch_stack(v_pred, M_c); vs = quad_batch_stack(v_pred, M_s)
    vv = quad_batch_stack(v_pred, M_v)

    h_p = (pg_pred @ C_g.T) - Pd_batch - vp
    h_q = (qg_pred @ C_g.T) - Qd_batch - vq
    g_sf = (pf**2 + qf**2) - smax**2
    g_st = (pt**2 + qt**2) - smax**2
    g_ang_min = torch.tan(angmin) * vc - vs
    g_ang_max = vs - torch.tan(angmax) * vc
    g_v_max = vv - (Vmax**2)
    g_v_min = (Vmin**2) - vv

    cost_per_gen = c2 * (pg_pred ** 2) + c1 * pg_pred + c0
    obj = cost_per_gen.sum(dim=1).mean()

    # --- UPDATED TO MATCH BASELINE WEIGHT STRUCTURE ---
    loss_eq_p = h_p.pow(2).mean()
    loss_eq_q = h_q.pow(2).mean()
    
    loss_ineq = (
        F.relu(g_sf).pow(2).mean() + F.relu(g_st).pow(2).mean() +
        F.relu(g_ang_min).pow(2).mean() + F.relu(g_ang_max).pow(2).mean() +
        F.relu(g_v_max).pow(2).mean() + F.relu(g_v_min).pow(2).mean()
    )

    # --------------------------------------------------------
    # 4. DC3 TARGET LOSS
    # --------------------------------------------------------
    # Penalize distance between Neural Network output and the Repaired Target
    dc3_corr_loss = (
        F.mse_loss(v_pred, v_c.detach()) + 
        F.mse_loss(pg_pred, pg_c.detach()) + 
        F.mse_loss(qg_pred, qg_c.detach())
    )

    # --- UPDATED TOTAL LOSS CALCULATION ---
    total_loss = (
        (weights["primal_eq_p"] * loss_eq_p) + 
        (weights["primal_eq_q"] * loss_eq_q) + 
        (weights["primal_ineq"] * loss_ineq) + 
        (weights["obj"] * obj) + 
        (weights["dc3_corr"] * dc3_corr_loss)
    )

    # --------------------------------------------------------
    # DIAGNOSTICS FOR BENCHMARKING
    # --------------------------------------------------------
    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "loss_primal": (loss_eq_p + loss_eq_q + loss_ineq).detach().item(),
        "loss_dc3_corr": dc3_corr_loss.detach().item(),
        "obj_cost": obj.detach().item(),
        
        "max_h_p": h_p.abs().max().detach().item(),
        "max_h_q": h_q.abs().max().detach().item(),
        "max_thermal": torch.max(F.relu(g_sf).max(), F.relu(g_st).max()).detach().item(),
        "max_v_viol": torch.max(F.relu(g_v_max).max(), F.relu(g_v_min).max()).detach().item(),
        "max_gen_viol": 0.0 # Baseline model bounds generators by construction!
    }

    return total_loss, diagnostics


# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    # 0. Hardware Device Discovery & Optimization
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. Forcing exit to prevent CPU execution.")
    device = torch.device("cuda")
    print(f"Executing strictly on GPU: {torch.cuda.get_device_name(0)}")
    
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

    model_dc3 = BranchedBaselineQCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        slack_imag_idx=slack_imag_idx
    ).to(device)

    optimizer_dc3 = optim.Adam(model_dc3.parameters(), lr=1e-3)

    # --- UPDATED DC3 LOSS WEIGHTS ---
    loss_weights_dc3 = {
        "primal_eq_p": 1000.0,   # Matches baseline "eq_p"
        "primal_eq_q": 1000.0,   # Matches baseline "eq_q"
        "primal_ineq": 1.0,      # Soft penalty on inequality constraints
        "obj": 0.01,             # Generation cost weight
        "dc3_corr": 50.0         # Heavy weight pushing predictions towards the repaired targets
    }

    epochs = 10000
    # --- Initialize checkpoint trackers ---
    best_val_loss = float('inf')
    model_save_path = f"./model/best_dc3_MULTI_{case_name}_{epochs}epochs.pth"

    # 5. Optimization Loop Execution
    print("\nBeginning execution of parallelized training matrix loops for DC3...")
    start_time = time.time()
    for epoch in range(epochs):
        model_dc3.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer_dc3.zero_grad()
            
            # Inner loop configuration (corr_steps=5) for deep constraint completion
            loss, diag = compute_dc3_qcqp_smax_loss(
                model=model_dc3, 
                Pd_batch=Pd_batch, 
                Qd_batch=Qd_batch, 
                problem=problem, 
                weights=loss_weights_dc3,
                corr_steps=5,      
                corr_lr=1e-2      
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_dc3.parameters(), 10.0)
            optimizer_dc3.step()
            
        if epoch % 10 == 0:
            # 1. Switch to evaluation mode and freeze gradients
            model_dc3.eval()
            with torch.no_grad():
                # Evaluate the entire validation set at once
                val_loss, val_diag = compute_dc3_qcqp_smax_loss(model_dc3, val_Pd, val_Qd, problem, loss_weights_dc3)

            # 2. Checkpointing Logic: If this is the lowest validation loss we've seen, save it!
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_dc3.state_dict(), model_save_path)
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