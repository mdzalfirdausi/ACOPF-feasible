#!/usr/bin/env python3
"""
ACOPF Hard KKT (Unrolled Lagrangian Optimizer) Training Script
Optimized for CUDA Acceleration / Intel i7 Hybrid Architecture

Based on model_2_Hard_KKT.pdf: 
Predicts Primal & Duals -> Unrolls L_rho gradient descent -> Computes KKT Physics Losses
"""
import argparse
from datetime import datetime
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import os
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

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
        #       =       2B       +     2G         + 2B+2L+2L+2B+4G     
        #       =       6B       +     6G         + 4L
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

        # --- REPLACED VOLTAGE BOUNDING LOGIC ---
        # Split raw voltage outputs into Real and Imaginary parts
        vr_raw = v_raw[:, :self.nbus]
        vi_raw = v_raw[:, self.nbus:]

        Vmax_b = problem["Vmax"].reshape(1, -1).expand(B, -1) if hasattr(problem["Vmax"], "reshape") else problem["Vmax"].unsqueeze(0).expand(B, -1)
        Vmin_b = problem["Vmin"].reshape(1, -1).expand(B, -1) if hasattr(problem["Vmin"], "reshape") else problem["Vmin"].unsqueeze(0).expand(B, -1)

        # 1. Bound Real Voltage strictly between [Vmin, Vmax] using Sigmoid (Centers around nominal 1.0 p.u.)
        vr = Vmin_b + torch.sigmoid(vr_raw) * (Vmax_b - Vmin_b)
        
        # 2. Bound Imaginary Voltage (Angle differences keep imaginary components small, e.g., [-0.5*Vmax, 0.5*Vmax])
        vi = torch.tanh(vi_raw) * (Vmax_b * 0.5)

        v = torch.cat([vr, vi], dim=-1)
        # ---------------------------------------
        v_0 = v

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

def compute_hard_kkt_loss(model, Pd_batch, Qd_batch, problem, weights, kkt_steps=5, kkt_lr=1e-3, rho=1.0):
    B = Pd_batch.shape[0]
    
    # --------------------------------------------------------
    # 1. FORWARD PASS (Primal Guess & Predicted Duals)
    # --------------------------------------------------------
    (v_0, pg_0, qg_0, lam_p, lam_q, mu_sf, mu_st, 
     mu_ang_max, mu_ang_min, mu_v_max, mu_v_min, 
     mu_pg_max, mu_pg_min, mu_qg_max, mu_qg_min) = model(Pd_batch, Qd_batch, problem)

    nbus = problem["nbus"]
    f = problem["fbus"]
    t = problem["tbus"]
    f_exp = f.unsqueeze(0).expand(B, -1)
    t_exp = t.unsqueeze(0).expand(B, -1)
    
    # Pre-expand fixed limits
    smax2 = problem["smax"]**2
    Vmax2 = problem["Vmax"]**2
    Vmin2 = problem["Vmin"]**2
    angmin = problem["angmin"].unsqueeze(0).expand(B, -1)
    angmax = problem["angmax"].unsqueeze(0).expand(B, -1)
    pmax = problem["pmax"].unsqueeze(0).expand(B, -1)
    pmin = problem["pmin"].unsqueeze(0).expand(B, -1)
    qmax = problem["qmax"].unsqueeze(0).expand(B, -1)
    qmin = problem["qmin"].unsqueeze(0).expand(B, -1)

    # --- Nested Helper to Evaluate Physics for Graph-Incidence ---
    def evaluate_physics(v_curr, pg_curr, qg_curr):
        vr = v_curr[:, :nbus]; vi = v_curr[:, nbus:]
        vr_f = vr[:, f]; vi_f = vi[:, f]
        vr_t = vr[:, t]; vi_t = vi[:, t]
        
        vv_f = vr_f**2 + vi_f**2
        vv_t = vr_t**2 + vi_t**2
        v_rt_cross = vr_f * vr_t + vi_f * vi_t
        v_it_cross = vr_f * vi_t - vi_f * vr_t
        
        # 1D Branch Flows
        pf = problem["g11"] * vv_f - (problem["g12"] - problem["b21"]) * v_rt_cross + (problem["g21"] + problem["b12"]) * v_it_cross
        qf = -problem["b11"] * vv_f + (problem["b12"] + problem["g21"]) * v_rt_cross + (problem["b21"] - problem["g12"]) * v_it_cross
        pt = problem["g22"] * vv_t - (problem["g12"] + problem["b21"]) * v_rt_cross + (problem["g21"] - problem["b12"]) * v_it_cross
        qt = -problem["b22"] * vv_t + (problem["b12"] - problem["g21"]) * v_rt_cross - (problem["b21"] + problem["g12"]) * v_it_cross
        
        # Nodal Injections via scatter_add
        vp = problem["Gs"] * (vr**2 + vi**2)
        vq = -problem["Bs"] * (vr**2 + vi**2)
        vp = vp.scatter_add(1, f_exp, pf)
        vp = vp.scatter_add(1, t_exp, pt)
        vq = vq.scatter_add(1, f_exp, qf)
        vq = vq.scatter_add(1, t_exp, qt)

        h_p_out = (pg_curr @ problem["C_g"].T) - Pd_batch - vp
        h_q_out = (qg_curr @ problem["C_g"].T) - Qd_batch - vq
        g_sf_out = (pf**2 + qf**2) - smax2
        g_st_out = (pt**2 + qt**2) - smax2
        g_ang_min_out = torch.tan(angmin) * v_rt_cross - v_it_cross
        g_ang_max_out = v_it_cross - torch.tan(angmax) * v_rt_cross
        
        vv = vr**2 + vi**2
        g_v_max_out = vv - Vmax2
        g_v_min_out = Vmin2 - vv
        g_pg_max_out = pg_curr - pmax
        g_pg_min_out = pmin - pg_curr
        g_qg_max_out = qg_curr - qmax
        g_qg_min_out = qmin - qg_curr
        
        c2 = problem["c2"].unsqueeze(0).expand(B, -1)
        c1 = problem["c1"].unsqueeze(0).expand(B, -1)
        c0 = problem["c0"].unsqueeze(0).expand(B, -1) if "c0" in problem else 0.0
        # NOTE: Returned as [B] array (not mean) for exact per-sample Lagrangian construction
        obj_cost = (c2 * (pg_curr ** 2) + c1 * pg_curr + c0).sum(dim=1)
        
        return (h_p_out, h_q_out, g_sf_out, g_st_out, g_ang_min_out, g_ang_max_out, 
                g_v_max_out, g_v_min_out, g_pg_max_out, g_pg_min_out, g_qg_max_out, g_qg_min_out, obj_cost)

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
            (h_p, h_q, g_sf, g_st, g_ang_min, g_ang_max, g_v_max, g_v_min, 
             g_pg_max, g_pg_min, g_qg_max, g_qg_min, obj) = evaluate_physics(v, pg, qg)
            
            # Construct Augmented Lagrangian (L_rho) using predicted duals
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
            
            # Compute Gradient & Take Update Step
            grad_v, grad_pg, grad_qg = torch.autograd.grad(
                L_rho.mean(), (v, pg, qg), create_graph=is_training, retain_graph=True
            )
            
            # CRITICAL SAFETY: Clamp Augmented Lagrangian gradients
            grad_v = torch.clamp(grad_v, -1.0, 1.0)
            grad_pg = torch.clamp(grad_pg, -1.0, 1.0)
            grad_qg = torch.clamp(grad_qg, -1.0, 1.0)
            
            v = v - kkt_lr * grad_v
            pg = pg - kkt_lr * grad_pg
            qg = qg - kkt_lr * grad_qg

    # --------------------------------------------------------
    # 3. KKT PHYSICS LAYER EVALUATION (Post-Optimization)
    # --------------------------------------------------------
    (h_p_f, h_q_f, g_sf_f, g_st_f, g_ang_min_f, g_ang_max_f, g_v_max_f, g_v_min_f, 
     g_pg_max_f, g_pg_min_f, g_qg_max_f, g_qg_min_f, obj_f) = evaluate_physics(v, pg, qg)

    loss_eq_p = h_p_f.pow(2).mean()
    loss_eq_q = h_q_f.pow(2).mean()

    loss_ineq = (
        F.relu(g_sf_f).pow(2).mean() + F.relu(g_st_f).pow(2).mean() +
        F.relu(g_ang_min_f).pow(2).mean() + F.relu(g_ang_max_f).pow(2).mean() +
        F.relu(g_v_max_f).pow(2).mean() + F.relu(g_v_min_f).pow(2).mean() +
        F.relu(g_pg_max_f).pow(2).mean() + F.relu(g_pg_min_f).pow(2).mean() +
        F.relu(g_qg_max_f).pow(2).mean() + F.relu(g_qg_min_f).pow(2).mean()
    )

    cs_loss = (
        (mu_sf * g_sf_f).pow(2).mean() + (mu_st * g_st_f).pow(2).mean() +
        (mu_ang_max * g_ang_max_f).pow(2).mean() + (mu_ang_min * g_ang_min_f).pow(2).mean() +
        (mu_v_max * g_v_max_f).pow(2).mean() + (mu_v_min * g_v_min_f).pow(2).mean() +
        (mu_pg_max * g_pg_max_f).pow(2).mean() + (mu_pg_min * g_pg_min_f).pow(2).mean() +
        (mu_qg_max * g_qg_max_f).pow(2).mean() + (mu_qg_min * g_qg_min_f).pow(2).mean()
    )

    dual_feas_loss = (
        F.relu(-mu_sf).pow(2).mean() + F.relu(-mu_st).pow(2).mean() +
        F.relu(-mu_ang_max).pow(2).mean() + F.relu(-mu_ang_min).pow(2).mean() +
        F.relu(-mu_v_max).pow(2).mean() + F.relu(-mu_v_min).pow(2).mean() +
        F.relu(-mu_pg_max).pow(2).mean() + F.relu(-mu_pg_min).pow(2).mean() +
        F.relu(-mu_qg_max).pow(2).mean() + F.relu(-mu_qg_min).pow(2).mean()
    )

    final_obj = obj_f.mean()

    total_loss = (
        weights["primal_eq_p"] * loss_eq_p + 
        weights["primal_eq_q"] * loss_eq_q + 
        weights["primal_ineq"] * loss_ineq + 
        weights["cs"] * cs_loss +
        weights["dual_feas"] * dual_feas_loss +
        weights["obj"] * final_obj
    )

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
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="ACOPF Unsupervised Baseline PINN Training")
    parser.add_argument(
        "--case_name", 
        type=str, 
        required=True,
        help="Name of the grid case topology (without _<samples>.pt)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        required=True,
        help="Number of training epochs"
    )
    args = parser.parse_args()
    # 0. Hardware Device Discovery & Optimization 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA Hardware Acceleration Active: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        # 1. Check if running under SLURM allocation first
        if "SLURM_CPUS_PER_TASK" in os.environ:
            max_threads = int(os.environ["SLURM_CPUS_PER_TASK"])
        # 2. Check Linux cgroup process affinity (prevents oversubscription on shared nodes)
        elif hasattr(os, "sched_getaffinity"):
            max_threads = len(os.sched_getaffinity(0))
        # 3. Fallback to total physical/logical cores (Windows / Mac / Local execution)
        else:
            max_threads = os.cpu_count() or 1  # Fallback to 1 if detection fails

        torch.set_num_threads(max_threads)
        print(f"Running on CPU Profile. Adaptive thread threshold established at {max_threads} threads.")

    # 1. Load Data
    case_name = args.case_name
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
    
    train_Pd = problem["Pd_all"][:train_size].to(device=device, dtype=torch.float32)
    train_Qd = problem["Qd_all"][:train_size].to(device=device, dtype=torch.float32)
    val_Pd = problem["Pd_all"][train_size:train_size + val_size].to(device=device, dtype=torch.float32)
    val_Qd = problem["Qd_all"][train_size:train_size + val_size].to(device=device, dtype=torch.float32)

    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                problem[key] = value.to(device=device, dtype=torch.float32)
            else:
                problem[key] = value.to(device=device)

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

    model_kkt = torch.compile(model_kkt)
    optimizer_kkt = optim.Adam(model_kkt.parameters(), lr=1e-3)

    # --- HARD KKT LOSS WEIGHTS ---
    loss_weights_kkt = {
        "primal_eq_p": 1000.0,   
        "primal_eq_q": 1000.0,   
        "primal_ineq": 1000.0,   # Increased from 1.0 to stop thermal limit violations
        "obj": 0.0005,
        "cs": 10.0,              # Increased from 1.0 for tighter complementary slackness
        "dual_feas": 10.0        # Increased from 1.0
    }
    epochs = args.epochs
    best_val_loss = float('inf')
    # Generate a timestamp string like '20260702_090602' (YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"./model/best_hardkkt_{case_name}_{epochs}epochs_{timestamp}.pth"

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
                kkt_lr=1e-4,
                rho=1.0      
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_kkt.parameters(), 10.0)
            optimizer_kkt.step()
            
        if epoch % 100 == 0:  
            model_kkt.eval()
            with torch.no_grad():
                val_loss, val_diag = compute_hard_kkt_loss(model_kkt, val_Pd, val_Qd, problem, loss_weights_kkt)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model_kkt._orig_mod.state_dict() if hasattr(model_kkt, '_orig_mod') else model_kkt.state_dict()
                torch.save(state_dict, model_save_path)
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