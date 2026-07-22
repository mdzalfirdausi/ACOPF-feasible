#!/usr/bin/env python3
"""
ACOPF Unsupervised Rahul KKT PINN Training Script
Optimized for CUDA Acceleration / Intel i7 Hybrid Architecture
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

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
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

    def forward(self, Pd, Qd, problem):  # <-- ADD problem HERE
        B = Pd.shape[0]
        x = torch.cat([Pd, Qd], dim=-1)
        raw = self.net(x)
        
        # ----------------------------------------------------
        # 1. Slice & BOUND Primal Variables
        # ----------------------------------------------------
        idx = 0
        v_raw = raw[:, idx : idx + self.dim_v]; idx += self.dim_v
        pq_raw = raw[:, idx : idx + self.dim_g]; idx += self.dim_g
        
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
        
        # Structurally lock generation between [min, max] using Sigmoid
        # This makes negative generation (and negative costs) 100% IMPOSSIBLE!
        pmax_b = problem["pmax"].unsqueeze(0).expand(B, -1)
        pmin_b = problem["pmin"].unsqueeze(0).expand(B, -1)
        qmax_b = problem["qmax"].unsqueeze(0).expand(B, -1)
        qmin_b = problem["qmin"].unsqueeze(0).expand(B, -1)

        pg = pmin_b + torch.sigmoid(pq_raw[:, :self.ngen]) * (pmax_b - pmin_b)
        qg = qmin_b + torch.sigmoid(pq_raw[:, self.ngen:]) * (qmax_b - qmin_b)
        
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

def compute_rahul_kkt_smax_loss(model, Pd_batch, Qd_batch, problem, weights):
    B = Pd_batch.shape[0]
    
    # Forward Pass
    (v, pg, qg, lam_p, lam_q, mu_sf, mu_st, 
     mu_ang_max, mu_ang_min, mu_v_max, mu_v_min, 
     mu_pg_max, mu_pg_min, mu_qg_max, mu_qg_min) = model(Pd_batch, Qd_batch, problem) 

    # --------------------------------------------------------
    # A. PRIMAL EVALUATIONS (Graph / Branch-Incidence)
    # --------------------------------------------------------
    nbus = problem["nbus"]
    f = problem["fbus"]
    t = problem["tbus"]
    
    vr = v[:, :nbus]
    vi = v[:, nbus:]
    
    # Extract voltages at connected buses
    vr_f = vr[:, f]; vi_f = vi[:, f]
    vr_t = vr[:, t]; vi_t = vi[:, t]
    
    vv_f = vr_f**2 + vi_f**2
    vv_t = vr_t**2 + vi_t**2
    
    v_rt_cross = vr_f * vr_t + vi_f * vi_t
    v_it_cross = vr_f * vi_t - vi_f * vr_t
    
    # 1. 1D Branch Flows (Replaces all M_pf, M_qf, M_pt, M_qt dense contractions)
    pf = problem["g11"] * vv_f - (problem["g12"] - problem["b21"]) * v_rt_cross + (problem["g21"] + problem["b12"]) * v_it_cross
    qf = -problem["b11"] * vv_f + (problem["b12"] + problem["g21"]) * v_rt_cross + (problem["b21"] - problem["g12"]) * v_it_cross
    pt = problem["g22"] * vv_t - (problem["g12"] + problem["b21"]) * v_rt_cross + (problem["g21"] - problem["b12"]) * v_it_cross
    qt = -problem["b22"] * vv_t + (problem["b12"] - problem["g21"]) * v_rt_cross - (problem["b21"] + problem["g12"]) * v_it_cross
    
    # 2. Nodal Injections (Uses scatter_add for fast, zero-free accumulation)
    vp = problem["Gs"] * (vr**2 + vi**2)
    vq = -problem["Bs"] * (vr**2 + vi**2)
    
    f_exp = f.unsqueeze(0).expand(B, -1)
    t_exp = t.unsqueeze(0).expand(B, -1)
    
    vp = vp.scatter_add(1, f_exp, pf)
    vp = vp.scatter_add(1, t_exp, pt)
    
    vq = vq.scatter_add(1, f_exp, qf)
    vq = vq.scatter_add(1, t_exp, qt)
    
    # 3. Constraints
    h_p = (pg @ problem["C_g"].T) - Pd_batch - vp
    h_q = (qg @ problem["C_g"].T) - Qd_batch - vq
    
    g_sf = (pf**2 + qf**2) - problem["smax"]**2
    g_st = (pt**2 + qt**2) - problem["smax"]**2
    
    g_ang_min = torch.tan(problem["angmin"]) * v_rt_cross - v_it_cross
    g_ang_max = v_it_cross - torch.tan(problem["angmax"]) * v_rt_cross
    
    vv = vr**2 + vi**2
    g_v_max = vv - problem["Vmax"]**2
    g_v_min = problem["Vmin"]**2 - vv
    
    g_pg_max = pg - problem["pmax"]
    g_pg_min = problem["pmin"] - pg
    g_qg_max = qg - problem["qmax"]
    g_qg_min = problem["qmin"] - qg

    # --------------------------------------------------------
    # CALCULATE OBJECTIVE COST
    # --------------------------------------------------------
    c2 = problem["c2"].unsqueeze(0).expand(B, -1)
    c1 = problem["c1"].unsqueeze(0).expand(B, -1)
    c0 = problem["c0"].unsqueeze(0).expand(B, -1)
    
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
    # D. KKT STATIONARITY (Automatic Differentiation)
    # --------------------------------------------------------
    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()
    slack_ref_error = torch.abs(v[:, slack_imag_idx]).mean()

    # Apply scaling for pure unsupervised mode if Lg_Max exists
    lam_p_scaled = lam_p * problem["Lg_Max"][0] if "Lg_Max" in problem else lam_p
    mu_pg_max_scaled = mu_pg_max * problem["Lg_Max"][1] if "Lg_Max" in problem else mu_pg_max
    mu_pg_min_scaled = mu_pg_min * problem["Lg_Max"][2] if "Lg_Max" in problem else mu_pg_min

    # Build the exact Lagrangian Scalar
    L_lagrangian = (
        (lam_p_scaled * h_p).sum() + (lam_q * h_q).sum() +
        (mu_sf * g_sf).sum() + (mu_st * g_st).sum() +
        (mu_ang_max * g_ang_max).sum() + (mu_ang_min * g_ang_min).sum() +
        (mu_v_max * g_v_max).sum() + (mu_v_min * g_v_min).sum() +
        (mu_pg_max_scaled * g_pg_max).sum() + (mu_pg_min_scaled * g_pg_min).sum() +
        (mu_qg_max * g_qg_max).sum() + (mu_qg_min * g_qg_min).sum()
    )
    
    # Safely compute gradients ONLY during the training loop
    if torch.is_grad_enabled():
        # Let PyTorch dynamically map exact KKT derivatives for the graph constraints
        dL_dv, dL_dpg, dL_dqg = torch.autograd.grad(
            outputs=L_lagrangian, 
            inputs=(v, pg, qg), 
            create_graph=True
        )
        
        # Add generator objective cost derivative analytically
        dL_dpg_total = dL_dpg + (2 * c2 * pg) + c1
        dL_dqg_total = dL_dqg
        
        stationarity_loss = dL_dpg_total.pow(2).mean() + dL_dqg_total.pow(2).mean() + dL_dv.pow(2).mean()
    else:
        # Skip heavy 2nd-order derivatives during validation
        stationarity_loss = torch.tensor(0.0, device=v.device)

    # --------------------------------------------------------
    # E. PRIMAL LOSS
    # --------------------------------------------------------
    loss_eq_p = h_p.pow(2).mean()
    loss_eq_q = h_q.pow(2).mean()
    
    loss_ineq = (
        F.relu(g_sf).pow(2).mean() + F.relu(g_st).pow(2).mean() +
        F.relu(g_ang_min).pow(2).mean() + F.relu(g_ang_max).pow(2).mean() +
        F.relu(g_v_max).pow(2).mean() + F.relu(g_v_min).pow(2).mean() +
        F.relu(g_pg_max).pow(2).mean() + F.relu(g_pg_min).pow(2).mean() +
        F.relu(g_qg_max).pow(2).mean() + F.relu(g_qg_min).pow(2).mean()
    )

    # --------------------------------------------------------
    # F. TOTAL KKT LOSS AGGREGATION
    # --------------------------------------------------------
    total_loss = (
        weights["primal_eq_p"] * loss_eq_p +
        weights["primal_eq_q"] * loss_eq_q +
        weights["primal_ineq"] * loss_ineq +
        weights["cs"] * cs_loss +
        weights["dual_feas"] * dual_feas_loss +
        weights["stationarity"] * stationarity_loss +
        weights["slack_ref"] * slack_ref_error 
    )

    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "loss_primal": (loss_eq_p + loss_eq_q + loss_ineq).detach().item(),
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
        torch.set_num_threads(12)
        print("Running on CPU Profile. Thread threshold established at 12 loops.")

    # 1. Load Data
    case_name = args.case_name
    total_samples = 10000
    dataset_path = f'./dataset/{case_name}_{total_samples}.pt'
    
    try:
        problem = torch.load(dataset_path, map_location=device)
    except FileNotFoundError:
        print(f"CRITICAL: Admittance topology dataset not found at target: {dataset_path}")
        sys.exit(1)

    # 2. Extract Data Split Slices & Cast to Float32
    actual_total_samples = problem["Pd_all"].shape[0] 
    train_size = int(0.8 * actual_total_samples)
    val_size = int(0.1 * actual_total_samples)

    print(f"Problem Geometry Linked -> Matrix Samples: {actual_total_samples}")
    
    # Slice arrays, cast to float32, and deploy to target device
    train_Pd = problem["Pd_all"][:train_size].to(device=device, dtype=torch.float32)
    train_Qd = problem["Qd_all"][:train_size].to(device=device, dtype=torch.float32)
    val_Pd = problem["Pd_all"][train_size:train_size + val_size].to(device=device, dtype=torch.float32)
    val_Qd = problem["Qd_all"][train_size:train_size + val_size].to(device=device, dtype=torch.float32)

    # Transition background system tensors to matching device AND float32 precision
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

    # 4. Model Instantiation & Parameter Configurations
    model_rahul = RahulSinglePINN_Smax(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        nbranch=problem["nbranch"]
    ).to(device)
    model_rahul = torch.compile(model_rahul)
    loss_weights_rahul = {
    "primal_eq_p": 1000.0,   # Matches baseline "eq_p"
    "primal_eq_q": 1000.0,   # Matches baseline "eq_q"
    "slack_ref": 1000.0,   
    "primal_ineq": 1000.0,      # Matches baseline "thermal", "ang", "v"
    "cs": 10.0,               # KKT Complementary Slackness
    "dual_feas": 10.0,        # KKT Dual Feasibility
    "stationarity": 0.0005     # KKT Stationarity (Implicitly handles your objective cost)
    }

    optimizer_rahul = optim.Adam(model_rahul.parameters(), lr=1e-3)
    epochs = args.epochs
    # --- Initialize checkpoint trackers ---
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"./model/best_rahul_model_{case_name}_{epochs}epochs_{timestamp}.pth"

    # 5. Optimization Loop Execution
    print("\nBeginning execution of parallelized training matrix loops for Rahul KKT PINN...")
    start_time = time.time()
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
            
        if epoch % 100 == 0:  
            # 1. Switch to evaluation mode and freeze gradients
            model_rahul.eval()
            with torch.no_grad():
                # Evaluate the entire validation set at once
                val_loss, val_diag = compute_rahul_kkt_smax_loss(model_rahul, val_Pd, val_Qd, problem, loss_weights_rahul)

            # 2. Checkpointing Logic: If this is the lowest validation loss we've seen, save it!
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model_rahul.state_dict(), model_save_path)
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