#!/usr/bin/env python3
"""
ACOPF DC3 - Exact NIRARV Implementation
Predict Partial -> Complete Equalities -> Correct Inequalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time
from datetime import datetime

torch.set_default_dtype(torch.float32)

class PartialQCQPMLP(nn.Module):
    """
    TRUE DC3 PARTIAL PREDICTOR
    Only outputs the independent control variables:
    - Pg for non-slack generators
    - Vr, Vi for all generator buses
    """
    def __init__(self, nbus: int, ngen: int, slack_imag_idx: int, hidden: int = 512):
        super().__init__()
        self.nbus = nbus
        self.ngen = ngen
        self.slack_imag_idx = int(slack_imag_idx)
        
        self.in_dim = 2 * nbus
        # We only predict Pg for (ngen - 1) buses, and V for (ngen) buses
        self.out_dim = (self.ngen - 1) + (2 * self.ngen)
        
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, Pd: torch.Tensor, Qd: torch.Tensor, problem: dict) -> tuple:
        B = Pd.shape[0]
        x = torch.cat([Pd, Qd], dim=-1)
        raw = self.net(x)
        
        # 1. Extract Partial Voltages (Generators Only)
        v_gen_raw = raw[:, :2 * self.ngen]
        vr_gen_raw = v_gen_raw[:, :self.ngen]
        vi_gen_raw = v_gen_raw[:, self.ngen:]
        
        vr_gen = torch.sigmoid(vr_gen_raw) * 2.0  # Scaled safely around 1.0 p.u.
        vi_gen = torch.tanh(vi_gen_raw) * 0.5
        
        # 2. Extract Partial Generation (Non-Slack Only)
        pg_pv_raw = raw[:, 2 * self.ngen:]
        
        # DYNAMIC INDEXING FIX: Use C_g matrix to find the exact generator array indices
        idx_pv = problem["pv"].long()
        idx_pv_gen = problem["C_g"][idx_pv].nonzero(as_tuple=True)[1]
        
        pmax_pv = problem["pmax"][idx_pv_gen].unsqueeze(0).expand(B, -1)
        pmin_pv = problem["pmin"][idx_pv_gen].unsqueeze(0).expand(B, -1)
        
        # Scale to problem bounds
        pg_pv = pmin_pv + torch.sigmoid(pg_pv_raw) * (pmax_pv - pmin_pv)
        
        return vr_gen, vi_gen, pg_pv

class QCQP_Completion_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vr_gen, vi_gen, pg_pv, Pd, Qd, problem):
        B = vr_gen.shape[0]
        nbus = problem["nbus"]
        device = vr_gen.device
        
        # Safely cast bus indices
        idx_slack = problem["slack"].view(-1).long()
        idx_pv = problem["pv"].view(-1).long()
        idx_pq = problem["pq"].view(-1).long()
        idx_gen = torch.cat([idx_slack, idx_pv])
        
        # DYNAMIC INDEXING FIX: Use C_g matrix to safely find generator array indices
        idx_slack_gen = problem["C_g"][idx_slack].nonzero(as_tuple=True)[1]
        idx_pv_gen = problem["C_g"][idx_pv].nonzero(as_tuple=True)[1]
        
        # 1. INITIALIZE FULL STATE VECTORS
        v_full = torch.ones(B, 2 * nbus, device=device)
        v_full[:, nbus:] = 0.0 # Imaginary voltages start at 0
        
        # Inject predictions into known slots
        v_full[:, idx_gen] = vr_gen
        v_full[:, idx_gen + nbus] = vi_gen
        
        # Identify Unknowns: vr and vi at PQ (Load) buses
        unknown_v_idx = torch.cat([idx_pq, idx_pq + nbus])
        known_v_idx = torch.cat([idx_gen, idx_gen + nbus])
        
        # 2. NEWTON-RAPHSON LOOP
        Mp, Mq = problem["M_p"], problem["M_q"]
        J_inv = None
        
        for _ in range(50): # 50 iterations matches original DC3 PFFunction
            # Evaluate QCQP Nodal Injections
            vp = torch.einsum('bi, nij, bj -> bn', v_full, Mp, v_full)
            vq = torch.einsum('bi, nij, bj -> bn', v_full, Mq, v_full)
            
            # Calculate mismatch at Load (PQ) buses
            h_p_pq = -Pd[:, idx_pq] - vp[:, idx_pq]
            h_q_pq = -Qd[:, idx_pq] - vq[:, idx_pq]
            mismatch = torch.cat([h_p_pq, h_q_pq], dim=1) 
            
            if mismatch.abs().max() < 1e-4:
                break
                
            # Build Exact Jacobian (Derivative of v^T M v is 2 M v)
            J_P_full = 2 * torch.einsum('nij, bj -> bni', Mp, v_full)
            J_Q_full = 2 * torch.einsum('nij, bj -> bni', Mq, v_full)
            
            # Slice Jacobian for (Equations at PQ) x (Unknowns at PQ)
            J_P_sub = J_P_full[:, idx_pq, :][:, :, unknown_v_idx]
            J_Q_sub = J_Q_full[:, idx_pq, :][:, :, unknown_v_idx]
            J = torch.cat([J_P_sub, J_Q_sub], dim=1) 
            
            # Newton Step
            J_inv = torch.linalg.inv(J)
            delta = torch.bmm(J_inv, mismatch.unsqueeze(-1)).squeeze(-1)
            v_full[:, unknown_v_idx] += delta
            
        # 3. DIRECT SOLVE (Dependent Variables)
        vp_final = torch.einsum('bi, nij, bj -> bn', v_full, Mp, v_full)
        vq_final = torch.einsum('bi, nij, bj -> bn', v_full, Mq, v_full)
        
        pg_full = torch.zeros(B, problem["ngen"], device=device)
        pg_full[:, idx_pv_gen] = pg_pv 
        
        pg_slack = Pd[:, idx_slack] + vp_final[:, idx_slack]
        pg_full[:, idx_slack_gen] = pg_slack.squeeze(-1)
        
        qg_full = Qd[:, idx_gen] + vq_final[:, idx_gen]

        # Save context for Backward Pass (Implicit Function Theorem)
        ctx.save_for_backward(J_inv, v_full, unknown_v_idx, known_v_idx, Mp, Mq, idx_pq, idx_pv_gen)
        
        return v_full, pg_full, qg_full

    @staticmethod
    def backward(ctx, grad_v_full, grad_pg_full, grad_qg_full):
        J_inv, v_full, unknown_v_idx, known_v_idx, Mp, Mq, idx_pq, idx_pv_gen = ctx.saved_tensors
        
        # IMPLICIT FUNCTION THEOREM (Backpropagate through the solver)
        grad_unknowns = grad_v_full[:, unknown_v_idx]
        d_int = torch.bmm(J_inv.transpose(1, 2), grad_unknowns.unsqueeze(-1)).squeeze(-1)
        
        J_P_full = 2 * torch.einsum('nij, bj -> bni', Mp, v_full)
        J_Q_full = 2 * torch.einsum('nij, bj -> bni', Mq, v_full)
        J_P_known = J_P_full[:, idx_pq, :][:, :, known_v_idx]
        J_Q_known = J_Q_full[:, idx_pq, :][:, :, known_v_idx]
        J_known = torch.cat([J_P_known, J_Q_known], dim=1) 
        
        grad_v_gen_combined = -torch.bmm(J_known.transpose(1, 2), d_int.unsqueeze(-1)).squeeze(-1)
        
        ngen = grad_v_gen_combined.shape[1] // 2
        grad_vr_gen = grad_v_gen_combined[:, :ngen]
        grad_vi_gen = grad_v_gen_combined[:, ngen:]
        
        grad_pg_pv = grad_pg_full[:, idx_pv_gen] 
        
        return grad_vr_gen, grad_vi_gen, grad_pg_pv, None, None, None

def evaluate_inequalities(v_full, pg_full, qg_full, problem):
    """Helper to extract physical inequalities for both correction and final loss"""
    B = v_full.shape[0]
    nbus = problem["nbus"]
    fbus = problem["fbus"].long()
    tbus = problem["tbus"].long()

    # Reconstruct branch flows
    vr_f, vi_f = v_full[:, fbus], v_full[:, fbus + nbus]
    vr_t, vi_t = v_full[:, tbus], v_full[:, tbus + nbus]
    
    vv_f = vr_f**2 + vi_f**2
    vv_t = vr_t**2 + vi_t**2
    v_rt_cross = vr_f * vr_t + vi_f * vi_t
    v_it_cross = vr_f * vi_t - vi_f * vr_t
    
    pf = problem["g11"] * vv_f - (problem["g12"] - problem["b21"]) * v_rt_cross + (problem["g21"] + problem["b12"]) * v_it_cross
    qf = -problem["b11"] * vv_f + (problem["b12"] + problem["g21"]) * v_rt_cross + (problem["b21"] - problem["g12"]) * v_it_cross
    pt = problem["g22"] * vv_t - (problem["g12"] + problem["b21"]) * v_rt_cross + (problem["g21"] - problem["b12"]) * v_it_cross
    qt = -problem["b22"] * vv_t + (problem["b12"] - problem["g21"]) * v_rt_cross - (problem["b21"] + problem["g12"]) * v_it_cross
    
    smax = problem["smax"].unsqueeze(0).expand(B, -1)
    g_sf = (pf**2 + qf**2) - smax**2
    g_st = (pt**2 + qt**2) - smax**2
    loss_thermal = F.relu(g_sf).pow(2).mean() + F.relu(g_st).pow(2).mean()
    
    # Calculate Generator bounds
    pmax = problem["pmax"].unsqueeze(0).expand(B, -1)
    pmin = problem["pmin"].unsqueeze(0).expand(B, -1)
    qmax = problem["qmax"].unsqueeze(0).expand(B, -1)
    qmin = problem["qmin"].unsqueeze(0).expand(B, -1)
    
    loss_gen = F.relu(pg_full - pmax).pow(2).mean() + F.relu(pmin - pg_full).pow(2).mean() + \
               F.relu(qg_full - qmax).pow(2).mean() + F.relu(qmin - qg_full).pow(2).mean()
               
    # Calculate Voltage bounds
    vv_full = v_full[:, :nbus]**2 + v_full[:, nbus:]**2
    Vmax = problem["Vmax"].unsqueeze(0).expand(B, -1)
    Vmin = problem["Vmin"].unsqueeze(0).expand(B, -1)
    
    loss_volt = F.relu(vv_full - Vmax**2).pow(2).mean() + F.relu(Vmin**2 - vv_full).pow(2).mean()
    
    max_thermal = torch.max(F.relu(g_sf).max(), F.relu(g_st).max()).detach().item()
    
    return loss_thermal, loss_gen, loss_volt, max_thermal

def compute_true_dc3_loss(model, Pd_batch, Qd_batch, problem, weights, corr_steps=5, corr_lr=1e-4):
    B = Pd_batch.shape[0]
    
    # 1. PREDICT INDEPENDENT VARIABLES
    vr_gen, vi_gen, pg_pv = model(Pd_batch, Qd_batch, problem)
    
    # =======================================================
    # 2. EXACT CORRECTION (Gradient Steps on Inequalities)
    # =======================================================
    momentum_vr, momentum_vi, momentum_pg = 0, 0, 0
    beta = 0.5 # Momentum for correction procedure
    
    for _ in range(corr_steps):
        # We must complete the equalities to evaluate the inequalities
        v_full, pg_full, qg_full = QCQP_Completion_Fn.apply(vr_gen, vi_gen, pg_pv, Pd_batch, Qd_batch, problem)
        
        # Evaluate Inequalities
        loss_thermal, loss_gen, loss_volt, _ = evaluate_inequalities(v_full, pg_full, qg_full, problem)
        loss_ineq = loss_thermal + loss_gen + loss_volt
        
        if loss_ineq.item() < 1e-6:
            break
            
        # Get gradients of inequalities w.r.t the independent variables using PyTorch Autograd
        # create_graph=True allows the outer NN to backpropagate completely through this unrolled loop
        g_vr, g_vi, g_pg = torch.autograd.grad(
            loss_ineq, (vr_gen, vi_gen, pg_pv), 
            create_graph=True, retain_graph=True
        )
        
        # Update with momentum
        momentum_vr = corr_lr * g_vr + beta * momentum_vr
        momentum_vi = corr_lr * g_vi + beta * momentum_vi
        momentum_pg = corr_lr * g_pg + beta * momentum_pg
        
        vr_gen = vr_gen - momentum_vr
        vi_gen = vi_gen - momentum_vi
        pg_pv  = pg_pv  - momentum_pg

    # =======================================================
    # 3. FINAL COMPLETION & TASK LOSS
    # =======================================================
    v_full, pg_full, qg_full = QCQP_Completion_Fn.apply(vr_gen, vi_gen, pg_pv, Pd_batch, Qd_batch, problem)
    loss_thermal, loss_gen, loss_volt, max_thermal = evaluate_inequalities(v_full, pg_full, qg_full, problem)
    
    # Evaluate Objective Cost
    cost_per_gen = problem["c2"].unsqueeze(0).expand(B, -1) * (pg_full ** 2) + \
                   problem["c1"].unsqueeze(0).expand(B, -1) * pg_full + \
                   problem["c0"].unsqueeze(0).expand(B, -1)
    obj_cost = cost_per_gen.sum(dim=1).mean()
    
    # Combine losses
    total_task_loss = weights["thermal"] * loss_thermal + weights["v"] * (loss_gen + loss_volt) + weights["obj"] * obj_cost
    
    diagnostics = {
        "loss_total": total_task_loss.detach().item(),
        "obj_cost": obj_cost.detach().item(),
        "max_thermal": max_thermal
    }
    
    return total_task_loss, diagnostics

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="ACOPF Exact DC3 PINN Training")
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

    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            problem[key] = value.to(device)

    # 3. Setup Dataset Pipeline
    batch_size = 1024 
    train_dataset = TensorDataset(train_Pd, train_Qd)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Model Instantiation & Parameter Configurations 
    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()

    model_dc3 = PartialQCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        slack_imag_idx=slack_imag_idx
    ).to(device)

    optimizer_dc3 = optim.Adam(model_dc3.parameters(), lr=1e-3)

    loss_weights_dc3 = {
        "thermal": 1000.0,   
        "v": 1000.0,         
        "obj": 0.0005        
    }

    epochs = args.epochs
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"./model/best_dc3_new_model_{case_name}_{epochs}epochs_{timestamp}.pth"

    # 5. Optimization Loop Execution
    print("\nBeginning execution of EXACT DC3 implicit backpropagation loops...")
    start_time = time.time()
    for epoch in range(epochs):
        model_dc3.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer_dc3.zero_grad()
            
            loss, diag = compute_true_dc3_loss(
                model=model_dc3, 
                Pd_batch=Pd_batch, 
                Qd_batch=Qd_batch, 
                problem=problem, 
                weights=loss_weights_dc3,
                corr_steps=5,      # Unroll 5 correction steps
                corr_lr=1e-4       # Learning rate for the correction phase
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_dc3.parameters(), 10.0)
            optimizer_dc3.step()
            
        if epoch % 100 == 0:
            model_dc3.eval()
            with torch.no_grad():
                # During evaluation, we typically don't run correction steps to test pure NN inference
                val_loss, val_diag = compute_true_dc3_loss(model_dc3, val_Pd, val_Qd, problem, loss_weights_dc3, corr_steps=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model_dc3._orig_mod.state_dict() if hasattr(model_dc3, '_orig_mod') else model_dc3.state_dict()
                torch.save(state_dict, model_save_path)
                saved_flag = " [*SAVED BEST*]"
            else:
                saved_flag = ""

            print(f"Epoch {epoch:4d} | Val Loss: {val_loss:.4f} | Val Cost: {val_diag['obj_cost']:7.2f} | "
                  f"Val Max Thermal: {val_diag['max_thermal']:.4f}{saved_flag}")
                  
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours, remainder = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "="*50)
    print(f"Training Complete!")
    print(f"Total Training Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    print(f"Best model weights saved to: {model_save_path}")
    print("="*50 + "\n")