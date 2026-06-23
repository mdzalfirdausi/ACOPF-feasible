#!/usr/bin/env python3
"""
ACOPF Unsupervised Rahul KKT PINN Training Script
Optimized for CUDA Acceleration / Intel i7 Hybrid Architecture
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# --- MODEL DEFINITION ---
# IMPORTANT: Paste your exact RahulSinglePINN_Smax class definition here!
class RahulSinglePINN_Smax(nn.Module):
    """
    Version 2: Single Neural Network for all variables (Primal + Dual).
    """
    def __init__(self, nbus, ngen, nbranch, hidden_dim=512):
        super().__init__()
        self.nbus = nbus
        self.ngen = ngen
        self.nbranch = nbranch
        
        in_dim = 2 * nbus # Pd and Qd
        
        # Calculate total output dimension
        self.dim_v = 2 * nbus
        self.dim_g = 2 * ngen # pg and qg
        self.num_duals = (4 * nbus) + (4 * nbranch) + (4 * ngen)
        
        out_dim = self.dim_v + self.dim_g + self.num_duals
        
        # A SINGLE Neural Network for everything
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, Pd, Qd):
        x = torch.cat([Pd, Qd], dim=-1)
        
        # Single forward pass
        raw = self.net(x)
        
        # ----------------------------------------------------
        # 1. Slice Primal Variables (Unbounded)
        # ----------------------------------------------------
        idx = 0
        v = raw[:, idx : idx + self.dim_v]; idx += self.dim_v
        
        pq = raw[:, idx : idx + self.dim_g]; idx += self.dim_g
        pg = pq[:, :self.ngen]
        qg = pq[:, self.ngen:]
        
        # ----------------------------------------------------
        # 2. Slice Dual Variables (Lagrange Multipliers)
        # ----------------------------------------------------
        lam_p = raw[:, idx : idx+self.nbus]; idx += self.nbus
        lam_q = raw[:, idx : idx+self.nbus]; idx += self.nbus
        
        mu_sf = raw[:, idx : idx+self.nbranch]; idx += self.nbranch
        mu_st = raw[:, idx : idx+self.nbranch]; idx += self.nbranch
        
        mu_ang_max = raw[:, idx : idx+self.nbranch]; idx += self.nbranch
        mu_ang_min = raw[:, idx : idx+self.nbranch]; idx += self.nbranch
        
        mu_v_max = raw[:, idx : idx+self.nbus]; idx += self.nbus
        mu_v_min = raw[:, idx : idx+self.nbus]; idx += self.nbus
        
        mu_pg_max = raw[:, idx : idx+self.ngen]; idx += self.ngen
        mu_pg_min = raw[:, idx : idx+self.ngen]; idx += self.ngen
        mu_qg_max = raw[:, idx : idx+self.ngen]; idx += self.ngen
        mu_qg_min = raw[:, idx : idx+self.ngen]; idx += self.ngen
        
        return (v, pg, qg, lam_p, lam_q, mu_sf, mu_st, 
                mu_ang_max, mu_ang_min, mu_v_max, mu_v_min, 
                mu_pg_max, mu_pg_min, mu_qg_max, mu_qg_min)
    
# --- UTILS & LOSS FUNCTIONS ---
def quad_batch_stack(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    # v: [B, d], M: [K, d, d] -> [B, K]
    return torch.einsum("bi,kij,bj->bk", v, M, v)

def batch_Mv(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # M: [K, d, d], v: [B, d] -> [B, K, d]
    return torch.einsum('kij,bj->bki', M, v)

def compute_rahul_kkt_smax_loss(model, Pd_batch, Qd_batch, problem, weights):
    B = Pd_batch.shape[0]
    
    # Forward Pass
    (v, pg, qg, lam_p, lam_q, mu_sf, mu_st, 
     mu_ang_max, mu_ang_min, mu_v_max, mu_v_min, 
     mu_pg_max, mu_pg_min, mu_qg_max, mu_qg_min) = model(Pd_batch, Qd_batch)

    # Matrices & Limits
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

    # --------------------------------------------------------
    # A. PRIMAL EVALUATIONS
    # --------------------------------------------------------
    vp = quad_batch_stack(v, M_p); vq = quad_batch_stack(v, M_q)
    pf = quad_batch_stack(v, M_pf); qf = quad_batch_stack(v, M_qf)
    pt = quad_batch_stack(v, M_pt); qt = quad_batch_stack(v, M_qt)
    vc = quad_batch_stack(v, M_c); vs = quad_batch_stack(v, M_s)
    vv = quad_batch_stack(v, M_v)

    # Equations
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

    # --------------------------------------------------------
    # CALCULATE OBJECTIVE COST
    # --------------------------------------------------------
    cost_per_gen = c2 * (pg ** 2) + c1 * pg + c0
    obj = cost_per_gen.sum(dim=1).mean()

    # --------------------------------------------------------
    # B. COMPLEMENTARY SLACKNESS (mu * g == 0)
    # --------------------------------------------------------
    cs_loss = (
        (mu_sf * g_sf).pow(2).mean() + (mu_st * g_st).pow(2).mean() +
        (mu_ang_max * g_ang_max).pow(2).mean() + (mu_ang_min * g_ang_min).pow(2).mean() +
        (mu_v_max * g_v_max).pow(2).mean() + (mu_v_min * g_v_min).pow(2).mean() +
        (mu_pg_max * g_pg_max).pow(2).mean() + (mu_pg_min * g_pg_min).pow(2).mean() +
        (mu_qg_max * g_qg_max).pow(2).mean() + (mu_qg_min * g_qg_min).pow(2).mean()
    )

    # --------------------------------------------------------
    # C. DUAL FEASIBILITY (mu >= 0)
    # --------------------------------------------------------
    dual_feas_loss = (
        F.relu(-mu_sf).pow(2).mean() + F.relu(-mu_st).pow(2).mean() +
        F.relu(-mu_ang_max).pow(2).mean() + F.relu(-mu_ang_min).pow(2).mean() +
        F.relu(-mu_v_max).pow(2).mean() + F.relu(-mu_v_min).pow(2).mean() +
        F.relu(-mu_pg_max).pow(2).mean() + F.relu(-mu_pg_min).pow(2).mean() +
        F.relu(-mu_qg_max).pow(2).mean() + F.relu(-mu_qg_min).pow(2).mean()
    )

    # --------------------------------------------------------
    # D. KKT STATIONARITY (Analytical Gradients = 0)
    # --------------------------------------------------------
    dL_dpg = (2 * c2 * pg) + c1 + (lam_p @ C_g) + mu_pg_max - mu_pg_min
    dL_dqg = (lam_q @ C_g) + mu_qg_max - mu_qg_min

    # Exact Analytical Gradient w.r.t voltage (v)
    dL_dv_p = -2 * torch.einsum('bk,bki->bi', lam_p, batch_Mv(M_p, v))
    dL_dv_q = -2 * torch.einsum('bk,bki->bi', lam_q, batch_Mv(M_q, v))
    
    # The Quartic Analytical Gradient for smax: 4 * mu * (p * Mp*v + q * Mq*v)
    dL_dv_sf = 4 * torch.einsum('bk,bk,bki->bi', mu_sf, pf, batch_Mv(M_pf, v)) + \
               4 * torch.einsum('bk,bk,bki->bi', mu_sf, qf, batch_Mv(M_qf, v))
    dL_dv_st = 4 * torch.einsum('bk,bk,bki->bi', mu_st, pt, batch_Mv(M_pt, v)) + \
               4 * torch.einsum('bk,bk,bki->bi', mu_st, qt, batch_Mv(M_qt, v))
    
    dL_dv_vmax = 2 * torch.einsum('bk,bki->bi', mu_v_max, batch_Mv(M_v, v))
    dL_dv_vmin = -2 * torch.einsum('bk,bki->bi', mu_v_min, batch_Mv(M_v, v))
    
    M_s_v = batch_Mv(M_s, v); M_c_v = batch_Mv(M_c, v)
    t_max = torch.tan(angmax).unsqueeze(-1); t_min = torch.tan(angmin).unsqueeze(-1)
    
    dL_dv_angmax = torch.einsum('bk,bki->bi', mu_ang_max, 2 * M_s_v - 2 * t_max * M_c_v)
    dL_dv_angmin = torch.einsum('bk,bki->bi', mu_ang_min, 2 * t_min * M_c_v - 2 * M_s_v)

    dL_dv = dL_dv_p + dL_dv_q + dL_dv_sf + dL_dv_st + dL_dv_vmax + dL_dv_vmin + dL_dv_angmax + dL_dv_angmin
    
    stationarity_loss = dL_dpg.pow(2).mean() + dL_dqg.pow(2).mean() + dL_dv.pow(2).mean()

    # --------------------------------------------------------
    # E. PRIMAL LOSS (Actual Physical Violations)
    # --------------------------------------------------------
    primal_loss = (
        h_p.pow(2).mean() + h_q.pow(2).mean() +
        F.relu(g_sf).pow(2).mean() + F.relu(g_st).pow(2).mean() +
        F.relu(g_ang_min).pow(2).mean() + F.relu(g_ang_max).pow(2).mean() +
        F.relu(g_v_max).pow(2).mean() + F.relu(g_v_min).pow(2).mean() +
        F.relu(g_pg_max).pow(2).mean() + F.relu(g_pg_min).pow(2).mean() +
        F.relu(g_qg_max).pow(2).mean() + F.relu(g_qg_min).pow(2).mean()
    )

    total_loss = (
        weights["primal"] * primal_loss +
        weights["cs"] * cs_loss +
        weights["dual_feas"] * dual_feas_loss +
        weights["stationarity"] * stationarity_loss
    )

    # --------------------------------------------------------
    # DIAGNOSTICS FOR BENCHMARKING
    # --------------------------------------------------------
    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "loss_primal": primal_loss.detach().item(),
        "loss_kkt_stat": stationarity_loss.detach().item(),
        "loss_kkt_cs": cs_loss.detach().item(),
        
        "obj_cost": obj.detach().item(),
        "max_h_p": h_p.abs().max().detach().item(),
        "max_h_q": h_q.abs().max().detach().item(),
        "max_thermal": torch.max(F.relu(g_sf).max(), F.relu(g_st).max()).detach().item(),
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

    # Transition background system tensors to matching target device
    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            problem[key] = value.to(device)

    # 3. Setup Dataset Pipeline
    batch_size = 1024 
    train_dataset = TensorDataset(train_Pd, train_Qd)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Model Instantiation & Parameter Configurations
    model_rahul = RahulSinglePINN_Smax(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        nbranch=problem["nbranch"]
    ).to(device)

    loss_weights_rahul = {
        "primal": 10.0,         
        "cs": 1.0,              
        "dual_feas": 1.0,       
        "stationarity": 0.01     
    }

    optimizer_rahul = optim.Adam(model_rahul.parameters(), lr=1e-3)
    epochs = 10000

    # 5. Optimization Loop Execution
    print("\nBeginning execution of parallelized training matrix loops for Rahul KKT PINN...")
    for epoch in range(epochs):
        model_rahul.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer_rahul.zero_grad()
            
            loss, diag = compute_rahul_kkt_smax_loss(
                model=model_rahul, 
                Pd_batch=Pd_batch, 
                Qd_batch=Qd_batch, 
                problem=problem, 
                weights=loss_weights_rahul
            )
            
            loss.backward()
            
            # Critical: Clip gradients to prevent the quartic s_max derivatives from exploding
            torch.nn.utils.clip_grad_norm_(model_rahul.parameters(), 10.0)
            optimizer_rahul.step()
            
        if epoch % 10 == 0:  
            print(f"Epoch {epoch:4d} | Cost: {diag['obj_cost']:7.2f} | "
                  f"Max P-Miss: {diag['max_h_p']:.4f} | Max Q-Miss: {diag['max_h_q']:.4f} | "
                  f"Max Gen Viol: {diag['max_gen_viol']:.4f} | Max Thermal: {diag['max_thermal']:.4f}")

    # 6. Save Model Checkpoint
    print("\nTraining complete. Saving Rahul model weights...")
    model_save_path = f"./model/rahul_pinn_{case_name}_{epochs}epochs.pth"
    torch.save(model_rahul.state_dict(), model_save_path)
    print(f"Rahul Model successfully saved to: {model_save_path}")