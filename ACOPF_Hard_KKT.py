#!/usr/bin/env python3
"""
ACOPF Hard KKT (Unrolled Lagrangian Optimizer) Training Script
Optimized for CUDA Acceleration / Intel i7 Hybrid Architecture

Based on model_2_Hard_KKT.pdf: 
Predicts Primal & Duals -> Unrolls L_rho gradient descent -> Computes KKT Physics Losses
"""
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# --- MODEL DEFINITION ---
class HardKKT_QCQPMLP(nn.Module):
    """
    Input:
        Pd: [B, nbus], Qd: [B, nbus]
    Output:
        Primal Guess: v_0, pg_0, qg_0 
        Duals: lam_p, lam_q, mu_sf, mu_st, mu_ang_..., mu_v_..., mu_pg_..., mu_qg_...
    """
    def __init__(self, nbus: int, ngen: int, nbranch: int, slack_imag_idx: int, hidden: int = 512):
        super().__init__()
        self.nbus = nbus
        self.ngen = ngen
        self.nbranch = nbranch
        self.slack_imag_idx = int(slack_imag_idx)

        self.in_dim = 2 * nbus
        self.out_dim_v = 2 * nbus
        self.out_dim_g = 2 * ngen 
        
        # Dual multipliers: 2 Eq + 2 Thermal + 2 Angle + 2 Voltage + 4 Gen Bounds
        self.num_duals = (2 * nbus) + (2 * nbranch) + (2 * nbranch) + (2 * nbus) + (4 * ngen)
        
        out_dim = self.out_dim_v + self.out_dim_g + self.num_duals

        # Core MLP Matrix Layer Sequence
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, Pd: torch.Tensor, Qd: torch.Tensor, problem: dict) -> tuple:
        B = Pd.shape[0]
        x = torch.cat([Pd, Qd], dim=-1)
        raw = self.net(x)

        # ----------------------------------------------------
        # 1. Slice Primal Variables (Initial Guess y_0)
        # ----------------------------------------------------
        idx = 0
        v_raw = raw[:, idx : idx+self.out_dim_v]; idx += self.out_dim_v
        g_raw = raw[:, idx : idx+self.out_dim_g]; idx += self.out_dim_g

        pg_raw = g_raw[:, :self.ngen]
        qg_raw = g_raw[:, self.ngen:]

        # Bounding Primal Guesses structurally
        Vmax_b = problem["Vmax"].reshape(1, -1).expand(B, -1)
        Vmax_full = torch.cat([Vmax_b, Vmax_b], dim=-1)
        v_0 = torch.tanh(v_raw) * Vmax_full

        v_clone = v_0.clone()
        v_clone[:, self.slack_imag_idx] = 0.0
        v_0 = v_clone

        pmax_b = problem["pmax"].reshape(1, -1).expand(B, -1)
        pmin_b = problem["pmin"].reshape(1, -1).expand(B, -1)
        qmax_b = problem["qmax"].reshape(1, -1).expand(B, -1)
        qmin_b = problem["qmin"].reshape(1, -1).expand(B, -1)

        pg_0 = pmin_b + torch.sigmoid(pg_raw) * (pmax_b - pmin_b)
        qg_0 = qmin_b + torch.sigmoid(qg_raw) * (qmax_b - qmin_b)

        # ----------------------------------------------------
        # 2. Slice Dual Variables (Lagrange Multipliers lambda, mu)
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

        return (v_0, pg_0, qg_0, lam_p, lam_q, mu_sf, mu_st, 
                mu_ang_max, mu_ang_min, mu_v_max, mu_v_min, 
                mu_pg_max, mu_pg_min, mu_qg_max, mu_qg_min)


# --- UTILS & LOSS FUNCTIONS ---
def quad_batch_stack(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,kij,bj->bk", v, M, v)

def compute_hard_kkt_loss(model, Pd_batch, Qd_batch, problem, weights, kkt_steps=5, kkt_lr=1e-3, rho=1.0):
    B = Pd_batch.shape[0]
    
    # --------------------------------------------------------
    # 1. FORWARD PASS (Primal Guess & Predicted Duals)
    # --------------------------------------------------------
    (v_0, pg_0, qg_0, lam_p, lam_q, mu_sf, mu_st, 
     mu_ang_max, mu_ang_min, mu_v_max, mu_v_min, 
     mu_pg_max, mu_pg_min, mu_qg_max, mu_qg_min) = model(Pd_batch, Qd_batch, problem)

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
    # 2. UNROLLED KKT OPTIMIZER LAYER (Augmented Lagrangian)
    # --------------------------------------------------------
    is_training = torch.is_grad_enabled()

    if not is_training:
        v = v_0.detach().requires_grad_(True)
        pg = pg_0.detach().requires_grad_(True)
        qg = qg_0.detach().requires_grad_(True)
    else:
        v = v_0
        pg = pg_0
        qg = qg_0
    
    with torch.enable_grad(): 
        for _ in range(kkt_steps):
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
            
            # Objective Cost
            cost = c2 * (pg ** 2) + c1 * pg + c0
            obj = cost.sum(dim=1)

            # B. Construct Augmented Lagrangian (L_rho) using predicted duals
            lam_h = (lam_p * h_p).sum(dim=1) + (lam_q * h_q).sum(dim=1)
            
            mu_g = (mu_sf * g_sf).sum(dim=1) + (mu_st * g_st).sum(dim=1) + \
                   (mu_ang_max * g_ang_max).sum(dim=1) + (mu_ang_min * g_ang_min).sum(dim=1) + \
                   (mu_v_max * g_v_max).sum(dim=1) + (mu_v_min * g_v_min).sum(dim=1) + \
                   (mu_pg_max * g_pg_max).sum(dim=1) + (mu_pg_min * g_pg_min).sum(dim=1) + \
                   (mu_qg_max * g_qg_max).sum(dim=1) + (mu_qg_min * g_qg_min).sum(dim=1)
            
            penalty = h_p.pow(2).sum(dim=1) + h_q.pow(2).sum(dim=1) + \
                      F.relu(g_sf).pow(2).sum(dim=1) + F.relu(g_st).pow(2).sum(dim=1) + \
                      F.relu(g_ang_max).pow(2).sum(dim=1) + F.relu(g_ang_min).pow(2).sum(dim=1) + \
                      F.relu(g_v_max).pow(2).sum(dim=1) + F.relu(g_v_min).pow(2).sum(dim=1) + \
                      F.relu(g_pg_max).pow(2).sum(dim=1) + F.relu(g_pg_min).pow(2).sum(dim=1) + \
                      F.relu(g_qg_max).pow(2).sum(dim=1) + F.relu(g_qg_min).pow(2).sum(dim=1)

            L_rho = obj + lam_h + mu_g + (rho / 2.0) * penalty
            
            # C. Compute Gradient & Take Update Step
            grad_v, grad_pg, grad_qg = torch.autograd.grad(
                L_rho.mean(), (v, pg, qg), create_graph=is_training, retain_graph=True
            )
            
            v = v - kkt_lr * grad_v
            pg = pg - kkt_lr * grad_pg
            qg = qg - kkt_lr * grad_qg

    # --------------------------------------------------------
    # 3. KKT PHYSICS LAYER EVALUATION (Post-Optimization)
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

    # I. Primal Equality Loss (l_eq)
    loss_eq_p = h_p_f.pow(2).mean()
    loss_eq_q = h_q_f.pow(2).mean()

    # II. Primal Inequality Loss (l_ineq)
    loss_ineq = (
        F.relu(g_sf_f).pow(2).mean() + F.relu(g_st_f).pow(2).mean() +
        F.relu(g_ang_min_f).pow(2).mean() + F.relu(g_ang_max_f).pow(2).mean() +
        F.relu(g_v_max_f).pow(2).mean() + F.relu(g_v_min_f).pow(2).mean() +
        F.relu(g_pg_max_f).pow(2).mean() + F.relu(g_pg_min_f).pow(2).mean() +
        F.relu(g_qg_max_f).pow(2).mean() + F.relu(g_qg_min_f).pow(2).mean()
    )

    # III. Complementary Slackness Loss (l_comp)
    cs_loss = (
        (mu_sf * g_sf_f).pow(2).mean() + (mu_st * g_st_f).pow(2).mean() +
        (mu_ang_max * g_ang_max_f).pow(2).mean() + (mu_ang_min * g_ang_min_f).pow(2).mean() +
        (mu_v_max * g_v_max_f).pow(2).mean() + (mu_v_min * g_v_min_f).pow(2).mean() +
        (mu_pg_max * g_pg_max_f).pow(2).mean() + (mu_pg_min * g_pg_min_f).pow(2).mean() +
        (mu_qg_max * g_qg_max_f).pow(2).mean() + (mu_qg_min * g_qg_min_f).pow(2).mean()
    )

    # IV. Dual Feasibility Loss (l_dual) enforcing mu >= 0
    dual_feas_loss = (
        F.relu(-mu_sf).pow(2).mean() + F.relu(-mu_st).pow(2).mean() +
        F.relu(-mu_ang_max).pow(2).mean() + F.relu(-mu_ang_min).pow(2).mean() +
        F.relu(-mu_v_max).pow(2).mean() + F.relu(-mu_v_min).pow(2).mean() +
        F.relu(-mu_pg_max).pow(2).mean() + F.relu(-mu_pg_min).pow(2).mean() +
        F.relu(-mu_qg_max).pow(2).mean() + F.relu(-mu_qg_min).pow(2).mean()
    )

    # V. Objective Loss (l_obj)
    final_obj = (c2 * (pg ** 2) + c1 * pg + c0).sum(dim=1).mean()

    # --------------------------------------------------------
    # 4. TOTAL LOSS COMBINATION
    # --------------------------------------------------------
    total_loss = (
        weights["primal_eq_p"] * loss_eq_p + 
        weights["primal_eq_q"] * loss_eq_q + 
        weights["primal_ineq"] * loss_ineq + 
        weights["cs"] * cs_loss +
        weights["dual_feas"] * dual_feas_loss +
        weights["obj"] * final_obj
    )

    # Diagnostics
    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "loss_primal": (loss_eq_p + loss_eq_q + loss_ineq).detach().item(),
        "loss_cs": cs_loss.detach().item(),
        "obj_cost": final_obj.detach().item(),
        
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
    
    train_Pd = problem["Pd_all"][:train_size].to(device)
    train_Qd = problem["Qd_all"][:train_size].to(device)
    val_Pd = problem["Pd_all"][train_size:train_size + val_size].to(device)
    val_Qd = problem["Qd_all"][train_size:train_size + val_size].to(device)

    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            problem[key] = value.to(device)

    # 3. Setup Dataset Pipeline
    batch_size = 1024 
    train_dataset = TensorDataset(train_Pd, train_Qd)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Model Instantiation
    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()

    model_kkt = HardKKT_QCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        nbranch=problem["nbranch"],
        slack_imag_idx=slack_imag_idx
    ).to(device)

    optimizer_kkt = optim.Adam(model_kkt.parameters(), lr=1e-3)

    # --- HARD KKT LOSS WEIGHTS ---
    loss_weights_kkt = {
        "primal_eq_p": 1000.0,   
        "primal_eq_q": 1000.0,   
        "primal_ineq": 1.0,      
        "obj": 0.01,
        "cs": 1.0,               # Complementary slackness weight
        "dual_feas": 1.0         # Enforcing positive multipliers
    }

    epochs = 10000
    best_val_loss = float('inf')
    model_save_path = f"./model/best_hardkkt_model_{case_name}_{epochs}epochs.pth"

    # 5. Optimization Loop Execution
    print("\nBeginning execution of parallelized training matrix loops for Hard KKT...")
    start_time = time.time()
    for epoch in range(epochs):
        model_kkt.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer_kkt.zero_grad()
            
            # Run Hard KKT with 5 unrolled augmented lagrangian steps
            loss, diag = compute_hard_kkt_loss(
                model=model_kkt, 
                Pd_batch=Pd_batch, 
                Qd_batch=Qd_batch, 
                problem=problem, 
                weights=loss_weights_kkt,
                kkt_steps=5,     
                kkt_lr=1e-2,
                rho=1.0      
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_kkt.parameters(), 10.0)
            optimizer_kkt.step()
            
        if epoch % 10 == 0:  
            model_kkt.eval()
            with torch.no_grad():
                val_loss, val_diag = compute_hard_kkt_loss(model_kkt, val_Pd, val_Qd, problem, loss_weights_kkt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_kkt.state_dict(), model_save_path)
                saved_flag = " [*SAVED BEST*]"
            else:
                saved_flag = ""

            print(f"Epoch {epoch:4d} | Val Loss: {val_loss:.4f} | Val Cost: {val_diag['obj_cost']:7.2f} | "
                  f"Val Max P-Miss: {val_diag['max_h_p']:.4f} | Val Max Q-Miss: {val_diag['max_h_q']:.4f} | "
                  f"Val Max Gen Viol: {val_diag['max_gen_viol']:.4f} | Val Max Thermal: {val_diag['max_thermal']:.4f}{saved_flag}")
    
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours, remainder = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "="*50)
    print(f"Training Complete!")
    print(f"Total Training Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"Best model weights saved to: {model_save_path}")
    print("="*50 + "\n")