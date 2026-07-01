#!/usr/bin/env python3
"""
ACOPF Unsupervised Baseline PINN Training Script
Optimized for Intel i7-1255U / CUDA Acceleration
"""

import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# --- MODEL DEFINITION ---
class baselineQCQPMLP(nn.Module):
    """
    Input:
        Pd: [B, nbus]
        Qd: [B, nbus]
    Output:
        v:  [B, 2*nbus] (Rectangular voltages)
        pg: [B, ngen]   (Active generation)
        qg: [B, ngen]   (Reactive generation)
    """
    def __init__(self, nbus: int, ngen: int, slack_imag_idx: int, hidden: int = 512):
        super().__init__()
        self.nbus = nbus
        self.ngen = ngen
        self.in_dim = 2 * nbus
        self.out_dim_v = 2 * nbus
        self.out_dim_g = 2 * ngen 
        self.slack_imag_idx = int(slack_imag_idx)

        # Core MLP Matrix Layer Sequence
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.out_dim_v + self.out_dim_g),
        )

    def forward(self, Pd: torch.Tensor, Qd: torch.Tensor, problem: dict) -> tuple:
        B = Pd.shape[0]
        x = torch.cat([Pd, Qd], dim=-1)
        raw = self.net(x)

        # 1. Slice outputs
        v_raw = raw[:, :self.out_dim_v]
        g_raw = raw[:, self.out_dim_v:]
        
        pg_raw = g_raw[:, :self.ngen]
        qg_raw = g_raw[:, self.ngen:]

        # 2. Bound Voltages to [-Vmax, Vmax] using Tanh for smooth gradients
        Vmax_b = problem["Vmax"].reshape(1, -1).expand(B, -1)
        Vmax_full = torch.cat([Vmax_b, Vmax_b], dim=-1) # Real and imaginary spaces
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
def quad_batch_stack(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    # v: [B, d], M: [K, d, d] -> [B, K]
    return torch.einsum("bi,kij,bj->bk", v, M, v)

def compute_qcqp_loss(model: nn.Module, Pd_batch: torch.Tensor, Qd_batch: torch.Tensor, problem: dict, weights: dict):
    B = Pd_batch.shape[0]
    
    # Predict variables
    v, pg, qg = model(Pd_batch, Qd_batch, problem)

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
    c0 = problem["c0"].unsqueeze(0).expand(B, -1)

    # Evaluate Quadratic Forms
    vp = quad_batch_stack(v, M_p)
    vq = quad_batch_stack(v, M_q)
    
    pf = quad_batch_stack(v, M_pf)
    qf = quad_batch_stack(v, M_qf)
    pt = quad_batch_stack(v, M_pt)
    qt = quad_batch_stack(v, M_qt)
    
    vc = quad_batch_stack(v, M_c)
    vs = quad_batch_stack(v, M_s)
    vv = quad_batch_stack(v, M_v)

    # Objective (Eq 2a)
    cost_per_gen = c2 * (pg ** 2) + c1 * pg + c0
    obj = cost_per_gen.sum(dim=1).mean()
    
    # Branch Thermal Limits
    g_sf = (pf**2 + qf**2) - smax**2
    g_st = (pt**2 + qt**2) - smax**2
    
    # Nodal Power Balance
    h_p = (pg @ C_g.T) - Pd_batch - vp
    h_q = (qg @ C_g.T) - Qd_batch - vq
    
    g_pg_max = pg - pmax
    g_pg_min = pmin - pg
    g_qg_max = qg - qmax
    g_qg_min = qmin - qg

    # Angle Difference Stability
    g_ang_min = torch.tan(angmin) * vc - vs
    g_ang_max = vs - torch.tan(angmax) * vc

    # Voltage Magnitude Security
    g_v_max = vv - (Vmax**2)
    g_v_min = (Vmin**2) - vv

    # Compute Penalties
    loss_eq_p = h_p.pow(2).mean()
    loss_eq_q = h_q.pow(2).mean()
    
    loss_thermal = F.relu(g_sf).pow(2).mean() + F.relu(g_st).pow(2).mean()
    loss_ang = F.relu(g_ang_min).pow(2).mean() + F.relu(g_ang_max).pow(2).mean()
    loss_v = F.relu(g_v_max).pow(2).mean() + F.relu(g_v_min).pow(2).mean()

    total_loss = (
        weights["eq_p"] * loss_eq_p +
        weights["eq_q"] * loss_eq_q +
        weights["thermal"] * loss_thermal +
        weights["ang"] * loss_ang +
        weights["v"] * loss_v +
        weights["obj"] * obj
    )

    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "obj_cost": obj.detach().item(),
        "max_h_p": h_p.abs().max().detach().item(),
        "max_h_q": h_q.abs().max().detach().item(),
        "max_thermal": torch.max(F.relu(g_sf).max(), F.relu(g_st).max()).detach().item(),
        "max_v_viol": torch.max(F.relu(g_v_max).max(), F.relu(g_v_min).max()).detach().item(),
        "max_gen_viol": torch.max(
            torch.max(F.relu(g_pg_max).max(), F.relu(g_pg_min).max()),
            torch.max(F.relu(g_qg_max).max(), F.relu(g_qg_min).max())
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
        # Enforce execution thread optimization for hybrid i7-1255U architectures
        torch.set_num_threads(12)
        print("Running on CPU Profile. Thread threshold established at 12 loops.")

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

    model = baselineQCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        slack_imag_idx=slack_imag_idx
    ).to(device)

    loss_weights = {
        "eq_p": 1000.0,
        "eq_q": 1000.0,
        "thermal": 1.0,
        "ang": 1.0,
        "v": 1.0,
        "obj": 0.01
    }

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    # --- Initialize checkpoint trackers ---
    best_val_loss = float('inf')
    model_save_path = f"./model/best_pinn_model_{case_name}_{epochs}epochs.pth"

    # 5. Optimization Loop Execution
    start_time = time.time()
    print("\nBeginning execution of parallelized training matrix loops...")
    for epoch in range(epochs):
        model.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer.zero_grad()
            loss, diag = compute_qcqp_loss(model, Pd_batch, Qd_batch, problem, loss_weights)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        if epoch % 10 == 0:  
            # 1. Switch to evaluation mode and freeze gradients
            model.eval()
            with torch.no_grad():
                # Evaluate the entire validation set at once
                val_loss, val_diag = compute_qcqp_loss(model, val_Pd, val_Qd, problem, loss_weights)
            
            # 2. Checkpointing Logic: If this is the lowest validation loss we've seen, save it!
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                saved_flag = " [*SAVED BEST*]"
            else:
                saved_flag = ""

            # 3. Print the comparison
            print(f"Epoch {epoch:4d} | Val Loss: {val_loss:.4f} | Val Cost: {val_diag['obj_cost']:7.2f} | "
                  f"Val Max P-Miss: {val_diag['max_h_p']:.4f} | Max Q-Miss: {val_diag['max_h_q']:.4f} |" 
                  f" Val Max Gen Viol: {val_diag['max_gen_viol']:.4f} | Val Max Thermal: {val_diag['max_thermal']:.4f}{saved_flag}")
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