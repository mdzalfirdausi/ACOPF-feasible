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
        
        v_gen_raw = raw[:, :2 * self.ngen]
        vr_gen_raw = v_gen_raw[:, :self.ngen]
        vi_gen_raw = v_gen_raw[:, self.ngen:]
        
        vr_gen = torch.sigmoid(vr_gen_raw) * 2.0  
        vi_gen = torch.tanh(vi_gen_raw) * 0.5
        
        pg_pv_raw = raw[:, 2 * self.ngen:]
        
        idx_pv_gen = problem["pv_"].long()
        pmax_pv = problem["pmax"][idx_pv_gen].unsqueeze(0).expand(B, -1)
        pmin_pv = problem["pmin"][idx_pv_gen].unsqueeze(0).expand(B, -1)
        
        pg_pv = pmin_pv + torch.sigmoid(pg_pv_raw) * (pmax_pv - pmin_pv)
        
        return vr_gen, vi_gen, pg_pv

class QCQP_Completion_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vr_gen, vi_gen, pg_pv, Pd, Qd, problem):
        B = vr_gen.shape[0]
        nbus = problem["nbus"]
        device = vr_gen.device
        
        idx_slack = problem["slack"].view(-1).long()
        idx_pv = problem["pv"].view(-1).long()
        idx_pq = problem["pq"].view(-1).long()
        idx_gen = torch.cat([idx_slack, idx_pv])
        
        idx_slack_gen = problem["slack_"].view(-1).long()
        idx_pv_gen = problem["pv_"].view(-1).long()
        
        v_full = torch.ones(B, 2 * nbus, device=device)
        v_full[:, nbus:] = 0.0 
        
        v_full[:, idx_gen] = vr_gen
        v_full[:, idx_gen + nbus] = vi_gen
        
        unknown_v_idx = torch.cat([idx_pq, idx_pq + nbus])
        known_v_idx = torch.cat([idx_gen, idx_gen + nbus])
        
        Mp, Mq = problem["M_p"], problem["M_q"]
        J_inv = None
        
        for _ in range(50): 
            vp = torch.einsum('bi, nij, bj -> bn', v_full, Mp, v_full)
            vq = torch.einsum('bi, nij, bj -> bn', v_full, Mq, v_full)
            
            h_p_pq = -Pd[:, idx_pq] - vp[:, idx_pq]
            h_q_pq = -Qd[:, idx_pq] - vq[:, idx_pq]
            mismatch = torch.cat([h_p_pq, h_q_pq], dim=1) 
            
            if mismatch.abs().max() < 1e-4:
                break
                
            J_P_full = 2 * torch.einsum('nij, bj -> bni', Mp, v_full)
            J_Q_full = 2 * torch.einsum('nij, bj -> bni', Mq, v_full)
            
            J_P_sub = J_P_full[:, idx_pq, :][:, :, unknown_v_idx]
            J_Q_sub = J_Q_full[:, idx_pq, :][:, :, unknown_v_idx]
            J = torch.cat([J_P_sub, J_Q_sub], dim=1) 
            
            J_inv = torch.linalg.pinv(J)
            delta = torch.bmm(J_inv, mismatch.unsqueeze(-1)).squeeze(-1)
            v_full[:, unknown_v_idx] += delta
            
        vp_final = torch.einsum('bi, nij, bj -> bn', v_full, Mp, v_full)
        vq_final = torch.einsum('bi, nij, bj -> bn', v_full, Mq, v_full)
        
        pg_full = torch.zeros(B, problem["ngen"], device=device)
        pg_full[:, idx_pv_gen] = pg_pv 
        
        pg_slack = Pd[:, idx_slack] + vp_final[:, idx_slack]
        pg_full[:, idx_slack_gen] = pg_slack
        
        qg_full = Qd[:, idx_gen] + vq_final[:, idx_gen]

        ctx.save_for_backward(J_inv, v_full, unknown_v_idx, known_v_idx, Mp, Mq, idx_pq, idx_pv_gen)
        
        return v_full, pg_full, qg_full

    @staticmethod
    def backward(ctx, grad_v_full, grad_pg_full, grad_qg_full):
        J_inv, v_full, unknown_v_idx, known_v_idx, Mp, Mq, idx_pq, idx_pv_gen = ctx.saved_tensors
        
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
    B = v_full.shape[0]
    nbus = problem["nbus"]
    fbus = problem["fbus"].long()
    tbus = problem["tbus"].long()

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

    angmin = problem["angmin"].unsqueeze(0).expand(B, -1)
    angmax = problem["angmax"].unsqueeze(0).expand(B, -1)
    g_ang_min = torch.tan(angmin) * v_rt_cross - v_it_cross
    g_ang_max = v_it_cross - torch.tan(angmax) * v_rt_cross
    loss_ang = F.relu(g_ang_min).pow(2).mean() + F.relu(g_ang_max).pow(2).mean()
    
    pmax = problem["pmax"].unsqueeze(0).expand(B, -1)
    pmin = problem["pmin"].unsqueeze(0).expand(B, -1)
    qmax = problem["qmax"].unsqueeze(0).expand(B, -1)
    qmin = problem["qmin"].unsqueeze(0).expand(B, -1)
    
    g_pg_max = pg_full - pmax
    g_pg_min = pmin - pg_full
    g_qg_max = qg_full - qmax
    g_qg_min = qmin - qg_full
    
    loss_gen = F.relu(g_pg_max).pow(2).mean() + F.relu(g_pg_min).pow(2).mean() + \
               F.relu(g_qg_max).pow(2).mean() + F.relu(g_qg_min).pow(2).mean()
               
    vv_full = v_full[:, :nbus]**2 + v_full[:, nbus:]**2
    Vmax = problem["Vmax"].unsqueeze(0).expand(B, -1)
    Vmin = problem["Vmin"].unsqueeze(0).expand(B, -1)
    
    g_v_max = vv_full - Vmax**2
    g_v_min = Vmin**2 - vv_full
    
    loss_volt = F.relu(g_v_max).pow(2).mean() + F.relu(g_v_min).pow(2).mean()
    
    max_thermal = torch.max(F.relu(g_sf).max(), F.relu(g_st).max()).detach().item()
    max_gen_viol = torch.max(
        torch.max(F.relu(g_pg_max).max(), F.relu(g_pg_min).max()),
        torch.max(F.relu(g_qg_max).max(), F.relu(g_qg_min).max())
    ).detach().item()
    max_v_viol = torch.max(F.relu(g_v_max).max(), F.relu(g_v_min).max()).detach().item()
    
    return loss_thermal, loss_ang, loss_gen, loss_volt, max_thermal, max_gen_viol, max_v_viol

def compute_true_dc3_loss(model, Pd_batch, Qd_batch, problem, weights, corr_steps=5, corr_lr=1e-4):
    B = Pd_batch.shape[0]
    
    vr_gen, vi_gen, pg_pv = model(Pd_batch, Qd_batch, problem)
    
    momentum_vr, momentum_vi, momentum_pg = 0, 0, 0
    beta = 0.5 
    
    for _ in range(corr_steps):
        v_full, pg_full, qg_full = QCQP_Completion_Fn.apply(vr_gen, vi_gen, pg_pv, Pd_batch, Qd_batch, problem)
        loss_thermal, loss_ang, loss_gen, loss_volt, _, _, _ = evaluate_inequalities(v_full, pg_full, qg_full, problem)
        loss_ineq = loss_thermal + loss_ang + loss_gen + loss_volt
        
        if loss_ineq.item() < 1e-6:
            break
            
        g_vr, g_vi, g_pg = torch.autograd.grad(
            loss_ineq, (vr_gen, vi_gen, pg_pv), 
            create_graph=True, retain_graph=True
        )
        
        momentum_vr = corr_lr * g_vr + beta * momentum_vr
        momentum_vi = corr_lr * g_vi + beta * momentum_vi
        momentum_pg = corr_lr * g_pg + beta * momentum_pg
        
        vr_gen = vr_gen - momentum_vr
        vi_gen = vi_gen - momentum_vi
        pg_pv  = pg_pv  - momentum_pg

    v_full, pg_full, qg_full = QCQP_Completion_Fn.apply(vr_gen, vi_gen, pg_pv, Pd_batch, Qd_batch, problem)
    loss_thermal, loss_ang, loss_gen, loss_volt, max_thermal, max_gen_viol, max_v_viol = evaluate_inequalities(v_full, pg_full, qg_full, problem)
    
    cost_per_gen = problem["c2"].unsqueeze(0).expand(B, -1) * (pg_full ** 2) + \
                   problem["c1"].unsqueeze(0).expand(B, -1) * pg_full + \
                   problem["c0"].unsqueeze(0).expand(B, -1)
    obj_cost = cost_per_gen.sum(dim=1).mean()
    
    total_task_loss = weights["thermal"] * loss_thermal + weights["v"] * (loss_gen + loss_volt) + weights.get("ang", 1000.0) * loss_ang + weights["obj"] * obj_cost
    
    # Calculate Power Mismatches strictly for logging (the Newton solver makes these ~0)
    vp = torch.einsum('bi, nij, bj -> bn', v_full, problem["M_p"], v_full)
    vq = torch.einsum('bi, nij, bj -> bn', v_full, problem["M_q"], v_full)
    h_p = (pg_full @ problem["C_g"].T) - Pd_batch - vp
    h_q = (qg_full @ problem["C_g"].T) - Qd_batch - vq

    diagnostics = {
        "loss_total": total_task_loss.detach().item(),
        "obj_cost": obj_cost.detach().item(),
        "max_h_p": h_p.abs().max().detach().item(),
        "max_h_q": h_q.abs().max().detach().item(),
        "max_thermal": max_thermal,
        "max_gen_viol": max_gen_viol,
        "max_v_viol": max_v_viol
    }
    
    return total_task_loss, diagnostics

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
        import sys
        sys.exit(1)

    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            problem[key] = value.to(device)

    # =================================================================
    # DYNAMIC TOPOLOGY EXTRACTION & QCQP MATRIX BUILDER
    # =================================================================
    problem["M_p"], problem["M_q"] = build_qcqp_matrices(problem, device)
    
    C_g = problem["C_g"]
    
    # Find the slack bus safely by capturing the '1' in a_ref and mapping it to the bus index
    slack_idx_raw = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0]
    slack_bus_idx = slack_idx_raw % problem["nbus"]
    
    is_gen = C_g.sum(dim=1) > 0
    is_slack = torch.zeros_like(is_gen, dtype=torch.bool)
    is_slack[slack_bus_idx] = True
    
    is_pv = is_gen & (~is_slack)
    is_pq = ~is_gen
    
    problem["slack"] = is_slack.nonzero(as_tuple=True)[0]
    problem["pv"] = is_pv.nonzero(as_tuple=True)[0]
    problem["pq"] = is_pq.nonzero(as_tuple=True)[0]
    problem["slack_"] = C_g[problem["slack"]].nonzero(as_tuple=True)[1]
    problem["pv_"] = C_g[problem["pv"]].nonzero(as_tuple=True)[1]
    # =================================================================

    actual_total_samples = problem["Pd_all"].shape[0] 
    train_size = int(0.8 * actual_total_samples)
    val_size = int(0.1 * actual_total_samples)

    print(f"Problem Geometry Linked -> Matrix Samples: {actual_total_samples}")
    
    train_Pd = problem["Pd_all"][:train_size].to(device)
    train_Qd = problem["Qd_all"][:train_size].to(device)
    val_Pd = problem["Pd_all"][train_size:train_size + val_size].to(device)
    val_Qd = problem["Qd_all"][train_size:train_size + val_size].to(device)

    # 3. Setup Dataset Pipeline
    batch_size = 1024 
    train_dataset = TensorDataset(train_Pd, train_Qd)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Model Instantiation
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
        "ang": 1000.0,         
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
                corr_steps=5,      
                corr_lr=1e-4       
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_dc3.parameters(), 10.0)
            optimizer_dc3.step()
            
        if epoch % 100 == 0:
            model_dc3.eval()
            with torch.no_grad():
                val_loss, val_diag = compute_true_dc3_loss(model_dc3, val_Pd, val_Qd, problem, loss_weights_dc3, corr_steps=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model_dc3._orig_mod.state_dict() if hasattr(model_dc3, '_orig_mod') else model_dc3.state_dict()
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