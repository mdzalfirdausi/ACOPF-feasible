#!/usr/bin/env python3 
"""
ACOPF FSNet (Feasibility-Seeking Neural Network) Training Script
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
import os
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')

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

        # 3. Bound Generation strictly between [min, max] using Sigmoid
        pmax_b = problem["pmax"].reshape(1, -1).expand(B, -1)
        pmin_b = problem["pmin"].reshape(1, -1).expand(B, -1)
        qmax_b = problem["qmax"].reshape(1, -1).expand(B, -1)
        qmin_b = problem["qmin"].reshape(1, -1).expand(B, -1)

        pg = pmin_b + torch.sigmoid(pg_raw) * (pmax_b - pmin_b)
        qg = qmin_b + torch.sigmoid(qg_raw) * (qmax_b - qmin_b)

        return v, pg, qg

def quad_batch_stack(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,kij,bj->bk", v, M, v)

def hybrid_lbfgs_step(y_init, evaluate_fn, max_diff_iter=5, max_iter=20, memory=20):
    """
    Exact Hybrid L-BFGS solver matching the original FSNet implementation.
    Executes 'max_diff_iter' differentiable steps, then switches to non-differentiable 
    steps to save memory, before reconnecting the gradients.
    """
    y = y_init.clone()
    B, N = y.shape
    device = y.device
    
    S_hist = torch.zeros(memory, B, N, device=device)
    Y_hist = torch.zeros(memory, B, N, device=device)
    hist_len, hist_ptr = 0, 0
    
    is_training = torch.is_grad_enabled()
    f_val = evaluate_fn(y)
    g = torch.autograd.grad(f_val.sum(), y, create_graph=is_training)[0]
    
    y_diff = y.clone() 
    
    for k in range(max_iter):
        use_grad = is_training and (k < max_diff_iter)
        
        # Transition to non-differentiable phase (Truncated Backpropagation)
        if not use_grad and is_training and k == max_diff_iter:
            y_diff = y.clone() 
            y = y.detach().requires_grad_(True)
            g = g.detach()
            f_val = evaluate_fn(y)
            g = torch.autograd.grad(f_val.sum(), y, create_graph=False)[0]

        # Compute search direction (Two-loop recursion)
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            
            s_dot_y = (S[-1] * Y[-1]).sum(dim=1, keepdim=True)
            y_dot_y = (Y[-1] * Y[-1]).sum(dim=1, keepdim=True) + 1e-10
            gamma = s_dot_y / y_dot_y
            
            q = g.clone()
            alphas = []
            rho = 1.0 / ((S * Y).sum(dim=2, keepdim=True) + 1e-10)
            for i in range(hist_len - 1, -1, -1):
                alpha_i = rho[i] * (S[i] * q).sum(dim=1, keepdim=True)
                alphas.append(alpha_i)
                q = q - alpha_i * Y[i]
            
            r = gamma * q
            alphas = alphas[::-1]
            for i in range(hist_len):
                beta = rho[i] * (Y[i] * r).sum(dim=1, keepdim=True)
                r = r + S[i] * (alphas[i] - beta)
            d = -r
        else:
            d = -0.1 * g
        
        d = torch.clamp(d, min=-50.0, max=50.0)
        
        # Backtracking line search
        step = 1.0
        dir_deriv = (g * d).sum(dim=1)
        for _ in range(10):
            y_trial = y + step * d
            f_trial = evaluate_fn(y_trial)
            if (f_trial <= f_val + 1e-4 * step * dir_deriv).all():
                break
            step *= 0.5
            
        y_next = y + step * d
        f_next = evaluate_fn(y_next)
        g_next = torch.autograd.grad(f_next.sum(), y_next, create_graph=use_grad)[0]
        
        # Update history buffers
        if use_grad:
            S_hist[hist_ptr] = y_next - y
            Y_hist[hist_ptr] = g_next - g
        else:
            S_hist[hist_ptr] = (y_next - y).detach()
            Y_hist[hist_ptr] = (g_next - g).detach()

        hist_ptr = (hist_ptr + 1) % memory
        hist_len = min(hist_len + 1, memory)
        
        if use_grad:
            y, f_val, g = y_next, f_next, g_next
        else:
            y = y_next.detach().requires_grad_(True)
            f_val = f_next.detach()
            g = g_next.detach()
            
    # Reconnect the non-differentiable result to the computation graph
    if is_training and max_iter > max_diff_iter:
        return y_diff + (y - y_diff).detach()
    return y

def compute_fsnet_qcqp_smax_loss(model, Pd_batch, Qd_batch, problem, weights, seek_steps=5, seek_lr=1e-3):
    B = Pd_batch.shape[0]
    
    # --------------------------------------------------------
    # 1. FORWARD PASS (Initial Guess y_0)
    # --------------------------------------------------------
    v_0, pg_0, qg_0 = model(Pd_batch, Qd_batch, problem)

    nbus = problem["nbus"]
    f = problem["fbus"]
    t = problem["tbus"]
    
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

    def evaluate_physics(v_curr, pg_curr, qg_curr):
        vr = v_curr[:, :nbus]; vi = v_curr[:, nbus:]
        vr_f = vr[:, f]; vi_f = vi[:, f]
        vr_t = vr[:, t]; vi_t = vi[:, t]
        
        vv_f = vr_f**2 + vi_f**2
        vv_t = vr_t**2 + vi_t**2
        v_rt_cross = vr_f * vr_t + vi_f * vi_t
        v_it_cross = vr_f * vi_t - vi_f * vr_t
        
        pf = problem["g11"] * vv_f - (problem["g12"] - problem["b21"]) * v_rt_cross + (problem["g21"] + problem["b12"]) * v_it_cross
        qf = -problem["b11"] * vv_f + (problem["b12"] + problem["g21"]) * v_rt_cross + (problem["b21"] - problem["g12"]) * v_it_cross
        pt = problem["g22"] * vv_t - (problem["g12"] + problem["b21"]) * v_rt_cross + (problem["g21"] - problem["b12"]) * v_it_cross
        qt = -problem["b22"] * vv_t + (problem["b12"] - problem["g21"]) * v_rt_cross - (problem["b21"] + problem["g12"]) * v_it_cross
        
        # Nodal Injections via EXACT QCQP Matrices
        vp = torch.einsum('bi, nij, bj -> bn', v_curr, problem["M_p"], v_curr)
        vq = torch.einsum('bi, nij, bj -> bn', v_curr, problem["M_q"], v_curr)

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
        obj_cost = (c2 * (pg_curr ** 2) + c1 * pg_curr + c0).sum(dim=1).mean()
        
        return (h_p_out, h_q_out, g_sf_out, g_st_out, g_ang_min_out, g_ang_max_out, 
                g_v_max_out, g_v_min_out, g_pg_max_out, g_pg_min_out, g_qg_max_out, g_qg_min_out, obj_cost)

    # --------------------------------------------------------
    # 2. FSNET FEASIBILITY SEEKING (L-BFGS Loop)
    # --------------------------------------------------------
    def evaluate_violations(y_curr):
        v_curr = y_curr[:, :2*nbus]
        pg_curr = y_curr[:, 2*nbus: 2*nbus+problem["ngen"]]
        qg_curr = y_curr[:, 2*nbus+problem["ngen"]:]
        
        (h_p, h_q, g_sf, g_st, g_ang_min, g_ang_max, g_v_max, g_v_min, 
         g_pg_max, g_pg_min, g_qg_max, g_qg_min, _) = evaluate_physics(v_curr, pg_curr, qg_curr)
        
        viol_loss = (
            h_p.pow(2).sum(dim=1) + h_q.pow(2).sum(dim=1) +
            F.relu(g_sf).pow(2).sum(dim=1) + F.relu(g_st).pow(2).sum(dim=1) +
            F.relu(g_ang_min).pow(2).sum(dim=1) + F.relu(g_ang_max).pow(2).sum(dim=1) +
            F.relu(g_v_max).pow(2).sum(dim=1) + F.relu(g_v_min).pow(2).sum(dim=1) +
            F.relu(g_pg_max).pow(2).sum(dim=1) + F.relu(g_pg_min).pow(2).sum(dim=1) +
            F.relu(g_qg_max).pow(2).sum(dim=1) + F.relu(g_qg_min).pow(2).sum(dim=1)
        )
        return viol_loss
        
    y_init = torch.cat([v_0, pg_0, qg_0], dim=1)
    
    # Executes the Hybrid L-BFGS Solver
    y_final = hybrid_lbfgs_step(
        y_init, evaluate_violations, 
        max_diff_iter=seek_steps, 
        max_iter=20 # Matches FSNet max_iter default
    )
    
    v = y_final[:, :2*nbus]
    pg = y_final[:, 2*nbus: 2*nbus+problem["ngen"]]
    qg = y_final[:, 2*nbus+problem["ngen"]:]

    # --------------------------------------------------------
    # 3. FINAL TASK LOSS EVALUATION ON \hat{y} (Post-Seeking)
    # --------------------------------------------------------
    (h_p_f, h_q_f, g_sf_f, g_st_f, g_ang_min_f, g_ang_max_f, g_v_max_f, g_v_min_f, 
     g_pg_max_f, g_pg_min_f, g_qg_max_f, g_qg_min_f, obj) = evaluate_physics(v, pg, qg)

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
        (weights["obj"] * obj) +
        (50.0 * (F.mse_loss(v_0, v.detach()) + F.mse_loss(pg_0, pg.detach()) + F.mse_loss(qg_0, qg.detach())))
    )

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

def build_qcqp_matrices(problem, device):
    """Automatically constructs the exact QCQP Matrices (Mp, Mq) from 1D branches."""
    nbus = problem["nbus"]
    fbus = problem["fbus"].long()
    tbus = problem["tbus"].long()
    nline = fbus.shape[0]

    dtype = torch.float32

    Mp = torch.zeros((nbus, 2 * nbus, 2 * nbus), dtype=dtype, device=device)
    Mq = torch.zeros((nbus, 2 * nbus, 2 * nbus), dtype=dtype, device=device)

    Gs = problem["Gs"]
    Bs = problem["Bs"]
    idx = torch.arange(nbus, device=device)
    Mp[idx, idx, idx] = Gs
    Mp[idx, idx + nbus, idx + nbus] = Gs
    Mq[idx, idx, idx] = -Bs
    Mq[idx, idx + nbus, idx + nbus] = -Bs

    g11, g12, g21, g22 = problem["g11"], problem["g12"], problem["g21"], problem["g22"]
    b11, b12, b21, b22 = problem["b11"], problem["b12"], problem["b21"], problem["b22"]

    for l in range(nline):
        f, t = fbus[l], tbus[l]

        Mp[f, f, f] += g11[l]; Mp[f, f+nbus, f+nbus] += g11[l]
        c_rt = -0.5 * (g12[l] - b21[l])
        Mp[f, f, t] += c_rt; Mp[f, t, f] += c_rt
        Mp[f, f+nbus, t+nbus] += c_rt; Mp[f, t+nbus, f+nbus] += c_rt
        c_it = 0.5 * (g21[l] + b12[l])
        Mp[f, f, t+nbus] += c_it; Mp[f, t+nbus, f] += c_it
        Mp[f, f+nbus, t] -= c_it; Mp[f, t, f+nbus] -= c_it

        Mp[t, t, t] += g22[l]; Mp[t, t+nbus, t+nbus] += g22[l]
        c_rt_t = -0.5 * (g12[l] + b21[l])
        Mp[t, t, f] += c_rt_t; Mp[t, f, t] += c_rt_t
        Mp[t, t+nbus, f+nbus] += c_rt_t; Mp[t, f+nbus, t+nbus] += c_rt_t
        c_it_t = 0.5 * (g21[l] - b12[l])
        Mp[t, t, f+nbus] += c_it_t; Mp[t, f+nbus, t] += c_it_t
        Mp[t, t+nbus, f] -= c_it_t; Mp[t, f, t+nbus] -= c_it_t

        Mq[f, f, f] += -b11[l]; Mq[f, f+nbus, f+nbus] += -b11[l]
        c_rt_q = 0.5 * (b12[l] + g21[l])
        Mq[f, f, t] += c_rt_q; Mq[f, t, f] += c_rt_q
        Mq[f, f+nbus, t+nbus] += c_rt_q; Mq[f, t+nbus, f+nbus] += c_rt_q
        c_it_q = 0.5 * (b21[l] - g12[l])
        Mq[f, f, t+nbus] += c_it_q; Mq[f, t+nbus, f] += c_it_q
        Mq[f, f+nbus, t] -= c_it_q; Mq[f, t, f+nbus] -= c_it_q

        Mq[t, t, t] += -b22[l]; Mq[t, t+nbus, t+nbus] += -b22[l]
        c_rt_qt = 0.5 * (b12[l] - g21[l])
        Mq[t, t, f] += c_rt_qt; Mq[t, f, t] += c_rt_qt
        Mq[t, t+nbus, f+nbus] += c_rt_qt; Mq[t, f+nbus, t+nbus] += c_rt_qt
        c_it_qt = -0.5 * (b21[l] + g12[l])
        Mq[t, t, f+nbus] += c_it_qt; Mq[t, f+nbus, t] += c_it_qt
        Mq[t, t+nbus, f] -= c_it_qt; Mq[t, f, t+nbus] -= c_it_qt

    return Mp, Mq

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

    problem["M_p"], problem["M_q"] = build_qcqp_matrices(problem, device)

    # 3. Setup Dataset Pipeline
    batch_size = 1024 
    train_dataset = TensorDataset(train_Pd, train_Qd)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Model Instantiation & Parameter Configurations
    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()

    model_fsnet = baselineQCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        slack_imag_idx=slack_imag_idx
    ).to(device)
    model_fsnet = torch.compile(model_fsnet)
    optimizer_fsnet = optim.Adam(model_fsnet.parameters(), lr=1e-3)

    # --- UPDATED FSNET LOSS WEIGHTS ---
    loss_weights_fsnet = {
        "primal_eq_p": 1000.0,   # Matches baseline "eq_p"
        "primal_eq_q": 1000.0,   # Matches baseline "eq_q"
        "primal_ineq": 1000.0,      # Matches baseline inequalities
        "obj": 0.0005            # Generation cost weight matching baseline
    }

    epochs = args.epochs
    # --- Initialize checkpoint trackers ---
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"./model/best_fsnet_model_{case_name}_{epochs}epochs_{timestamp}.pth"

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
                seek_lr=1e-4      
            )
            
            loss.backward()
            
            # Clipping is mandatory because second-order autograd gradients can explode
            torch.nn.utils.clip_grad_norm_(model_fsnet.parameters(), 10.0)
            optimizer_fsnet.step()
            
        if epoch % 100 == 0:  
            # 1. Switch to evaluation mode and freeze gradients
            model_fsnet.eval()
            with torch.no_grad():
                # Evaluate the entire validation set at once
                val_loss, val_diag = compute_fsnet_qcqp_smax_loss(model_fsnet, val_Pd, val_Qd, problem, loss_weights_fsnet)

            # 2. Checkpointing Logic: If this is the lowest validation loss we've seen, save it!
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model_fsnet._orig_mod.state_dict() if hasattr(model_fsnet, '_orig_mod') else model_fsnet.state_dict()
                torch.save(state_dict, model_save_path)
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
