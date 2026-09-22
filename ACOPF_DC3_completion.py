#!/usr/bin/env python3
"""
ACOPF DC3 (Deep Constraint Completion & Correction) Training Script
Optimized for CUDA Acceleration / Intel i7 Hybrid Architecture
"""
import argparse
from datetime import datetime
import time
import sys
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
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
    
def compute_dc3_qcqp_smax_loss(
    model,
    Pd_batch,
    Qd_batch,
    problem,
    weights,
    completion_steps=10,
    completion_lr=1e-2,
    corr_steps=5,
    corr_lr=1e-2,
    completion_prox=1e-4,
    equality_preserve_weight=100.0,
):
    """
    QCQP-native DC3-style completion + correction.

    Stage 1 -- NN prediction:
        The common MLP predicts (v, pg, qg).

    Stage 2 -- equality completion:
        Keep pg fixed and numerically complete (v, qg) by minimizing only
        the AC active/reactive power-balance residuals.  A very small
        proximal term keeps the completed point close to the NN prediction.

    Stage 3 -- inequality correction:
        Starting from the completed point, reduce thermal, angle, and
        voltage-magnitude violations while strongly preserving the completed
        AC power-balance equalities.

    Stage 4 -- outer training loss:
        Train the MLP toward the final completed/corrected target while also
        retaining the common QCQP primal/objective losses.

    NOTE:
        This is a numerical completion layer adapted to the common rectangular
        QCQP output representation.  It is not the exact polar PV/PQ variable
        elimination used in the original DC3 ACOPF appendix.
    """
    B = Pd_batch.shape[0]

    # --------------------------------------------------------
    # 1. FORWARD PASS (Network Prediction)
    # --------------------------------------------------------
    v_pred, pg_pred, qg_pred = model(Pd_batch, Qd_batch, problem)

    nbus = problem["nbus"]
    f = problem["fbus"]
    t = problem["tbus"]
    f_exp = f.unsqueeze(0).expand(B, -1)
    t_exp = t.unsqueeze(0).expand(B, -1)

    smax2 = problem["smax"] ** 2
    Vmax2 = problem["Vmax"] ** 2
    Vmin2 = problem["Vmin"] ** 2
    angmin = problem["angmin"].unsqueeze(0).expand(B, -1)
    angmax = problem["angmax"].unsqueeze(0).expand(B, -1)

    def enforce_reference(v_curr):
        """Enforce Im(V_ref)=0 without changing tensor shape."""
        if model.slack_imag_idx is None:
            return v_curr
        mask = torch.ones_like(v_curr)
        mask[:, model.slack_imag_idx] = 0.0
        return v_curr * mask

    # --------------------------------------------------------
    # Physics evaluator
    # --------------------------------------------------------
    def evaluate_physics(v_curr, pg_curr, qg_curr):
        v_curr = enforce_reference(v_curr)

        vr = v_curr[:, :nbus]
        vi = v_curr[:, nbus:]
        vr_f = vr[:, f]
        vi_f = vi[:, f]
        vr_t = vr[:, t]
        vi_t = vi[:, t]

        vv_f = vr_f ** 2 + vi_f ** 2
        vv_t = vr_t ** 2 + vi_t ** 2
        v_rt_cross = vr_f * vr_t + vi_f * vi_t
        v_it_cross = vr_f * vi_t - vi_f * vr_t

        pf = (
            problem["g11"] * vv_f
            - (problem["g12"] - problem["b21"]) * v_rt_cross
            + (problem["g21"] + problem["b12"]) * v_it_cross
        )
        qf = (
            -problem["b11"] * vv_f
            + (problem["b12"] + problem["g21"]) * v_rt_cross
            + (problem["b21"] - problem["g12"]) * v_it_cross
        )
        pt = (
            problem["g22"] * vv_t
            - (problem["g12"] + problem["b21"]) * v_rt_cross
            + (problem["g21"] - problem["b12"]) * v_it_cross
        )
        qt = (
            -problem["b22"] * vv_t
            + (problem["b12"] - problem["g21"]) * v_rt_cross
            - (problem["b21"] + problem["g12"]) * v_it_cross
        )

        vp = problem["Gs"] * (vr ** 2 + vi ** 2)
        vq = -problem["Bs"] * (vr ** 2 + vi ** 2)
        vp = vp.scatter_add(1, f_exp, pf)
        vp = vp.scatter_add(1, t_exp, pt)
        vq = vq.scatter_add(1, f_exp, qf)
        vq = vq.scatter_add(1, t_exp, qt)

        h_p_out = (pg_curr @ problem["C_g"].T) - Pd_batch - vp
        h_q_out = (qg_curr @ problem["C_g"].T) - Qd_batch - vq

        g_sf_out = (pf ** 2 + qf ** 2) - smax2
        g_st_out = (pt ** 2 + qt ** 2) - smax2
        g_ang_min_out = torch.tan(angmin) * v_rt_cross - v_it_cross
        g_ang_max_out = v_it_cross - torch.tan(angmax) * v_rt_cross

        vv = vr ** 2 + vi ** 2
        g_v_max_out = vv - Vmax2
        g_v_min_out = Vmin2 - vv

        c2 = problem["c2"].unsqueeze(0).expand(B, -1)
        c1 = problem["c1"].unsqueeze(0).expand(B, -1)
        c0 = (
            problem["c0"].unsqueeze(0).expand(B, -1)
            if "c0" in problem
            else 0.0
        )
        obj_cost = (
            c2 * (pg_curr ** 2) + c1 * pg_curr + c0
        ).sum(dim=1).mean()

        return (
            h_p_out,
            h_q_out,
            g_sf_out,
            g_st_out,
            g_ang_min_out,
            g_ang_max_out,
            g_v_max_out,
            g_v_min_out,
            obj_cost,
        )

    # --------------------------------------------------------
    # 2. COMPLETION PHASE
    #    pg is the independent generation control.
    #    Complete v and qg using AC equality residuals only.
    # --------------------------------------------------------
    pg_comp = pg_pred.detach().clone()
    v_comp = enforce_reference(v_pred.detach().clone()).requires_grad_(True)
    qg_comp = qg_pred.detach().clone().requires_grad_(True)

    optimizer_comp = torch.optim.Adam(
        [v_comp, qg_comp],
        lr=completion_lr,
    )

    v_anchor = enforce_reference(v_pred.detach())
    qg_anchor = qg_pred.detach()

    with torch.enable_grad():
        for _ in range(completion_steps):
            optimizer_comp.zero_grad()

            (
                h_p_comp,
                h_q_comp,
                _, _, _, _, _, _, _,
            ) = evaluate_physics(v_comp, pg_comp, qg_comp)

            equality_loss = (
                h_p_comp.pow(2).mean()
                + h_q_comp.pow(2).mean()
            )

            # Select one nearby equality-completed point when the completion
            # system has multiple solutions.
            prox_loss = (
                F.mse_loss(enforce_reference(v_comp), v_anchor)
                + F.mse_loss(qg_comp, qg_anchor)
            )

            completion_loss = equality_loss + completion_prox * prox_loss
            completion_loss.backward()
            optimizer_comp.step()

            with torch.no_grad():
                # Enforce the reference-angle condition exactly.
                v_comp[:, model.slack_imag_idx] = 0.0

                # Keep completed reactive generation inside device bounds.
                qg_comp.clamp_(
                    min=problem["qmin"].reshape(1, -1),
                    max=problem["qmax"].reshape(1, -1),
                )

    # Completed point, detached from the inner optimizer.
    v_completed = enforce_reference(v_comp.detach())
    pg_completed = pg_comp.detach()
    qg_completed = qg_comp.detach()

    (
        h_p_completed,
        h_q_completed,
        _, _, _, _, _, _, _,
    ) = evaluate_physics(v_completed, pg_completed, qg_completed)

    completion_eq_loss = (
        h_p_completed.pow(2).mean()
        + h_q_completed.pow(2).mean()
    )

    # --------------------------------------------------------
    # 3. DC3 CORRECTION PHASE
    #    Start FROM the completed point.
    #    Reduce inequalities while preserving equalities.
    # --------------------------------------------------------
    v_c = v_completed.clone().requires_grad_(True)
    pg_c = pg_completed.clone().requires_grad_(True)
    qg_c = qg_completed.clone().requires_grad_(True)

    optimizer_corr = torch.optim.Adam(
        [v_c, pg_c, qg_c],
        lr=corr_lr,
    )

    with torch.enable_grad():
        for _ in range(corr_steps):
            optimizer_corr.zero_grad()

            (
                h_p_c,
                h_q_c,
                g_sf_c,
                g_st_c,
                g_ang_min_c,
                g_ang_max_c,
                g_v_max_c,
                g_v_min_c,
                _,
            ) = evaluate_physics(v_c, pg_c, qg_c)

            equality_preserve = (
                h_p_c.pow(2).mean()
                + h_q_c.pow(2).mean()
            )

            inequality_loss = (
                F.relu(g_sf_c).pow(2).mean()
                + F.relu(g_st_c).pow(2).mean()
                + F.relu(g_ang_min_c).pow(2).mean()
                + F.relu(g_ang_max_c).pow(2).mean()
                + F.relu(g_v_max_c).pow(2).mean()
                + F.relu(g_v_min_c).pow(2).mean()
            )

            corr_loss = (
                inequality_loss
                + equality_preserve_weight * equality_preserve
            )

            corr_loss.backward()
            optimizer_corr.step()

            with torch.no_grad():
                v_c[:, model.slack_imag_idx] = 0.0
                pg_c.clamp_(
                    min=problem["pmin"].reshape(1, -1),
                    max=problem["pmax"].reshape(1, -1),
                )
                qg_c.clamp_(
                    min=problem["qmin"].reshape(1, -1),
                    max=problem["qmax"].reshape(1, -1),
                )

    v_target = enforce_reference(v_c.detach())
    pg_target = pg_c.detach()
    qg_target = qg_c.detach()

    # --------------------------------------------------------
    # 4. STANDARD PRIMAL EVALUATION ON ORIGINAL NN OUTPUT
    # --------------------------------------------------------
    (
        h_p,
        h_q,
        g_sf,
        g_st,
        g_ang_min,
        g_ang_max,
        g_v_max,
        g_v_min,
        obj,
    ) = evaluate_physics(v_pred, pg_pred, qg_pred)

    loss_eq_p = h_p.pow(2).mean()
    loss_eq_q = h_q.pow(2).mean()

    loss_ineq = (
        F.relu(g_sf).pow(2).mean()
        + F.relu(g_st).pow(2).mean()
        + F.relu(g_ang_min).pow(2).mean()
        + F.relu(g_ang_max).pow(2).mean()
        + F.relu(g_v_max).pow(2).mean()
        + F.relu(g_v_min).pow(2).mean()
    )

    # --------------------------------------------------------
    # 5. COMPLETION/CORRECTION TARGET LOSS
    # --------------------------------------------------------
    dc3_corr_loss = (
        F.mse_loss(enforce_reference(v_pred), v_target)
        + F.mse_loss(pg_pred, pg_target)
        + F.mse_loss(qg_pred, qg_target)
    )

    total_loss = (
        weights["primal_eq_p"] * loss_eq_p
        + weights["primal_eq_q"] * loss_eq_q
        + weights["primal_ineq"] * loss_ineq
        + weights["obj"] * obj
        + weights["dc3_corr"] * dc3_corr_loss
    )

    diagnostics = {
        "loss_total": total_loss.detach().item(),
        "loss_primal": (loss_eq_p + loss_eq_q + loss_ineq).detach().item(),
        "loss_completion_eq": completion_eq_loss.detach().item(),
        "loss_dc3_corr": dc3_corr_loss.detach().item(),
        "obj_cost": obj.detach().item(),
        "max_h_p": h_p.abs().max().detach().item(),
        "max_h_q": h_q.abs().max().detach().item(),
        "max_h_p_completed": h_p_completed.abs().max().detach().item(),
        "max_h_q_completed": h_q_completed.abs().max().detach().item(),
        "max_thermal": torch.max(
            F.relu(g_sf).max(),
            F.relu(g_st).max(),
        ).detach().item(),
        "max_v_viol": torch.max(
            F.relu(g_v_max).max(),
            F.relu(g_v_min).max(),
        ).detach().item(),
        "max_gen_viol": 0.0,
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

    model_dc3 = baselineQCQPMLP(
        nbus=problem["nbus"],
        ngen=problem["ngen"],
        slack_imag_idx=slack_imag_idx
    ).to(device)

    optimizer_dc3 = optim.Adam(model_dc3.parameters(), lr=1e-3)

    # --- UPDATED DC3 LOSS WEIGHTS ---
    loss_weights_dc3 = {
        "primal_eq_p": 1000.0,   # Increased from 10.0
        "primal_eq_q": 1000.0,   # Increased from 10.0
        "primal_ineq": 1000.0,   # Increased from 1.0 to enforce thermal/voltage limits
        "obj": 0.0005,           
        "dc3_corr": 50.0         
    }

    epochs = args.epochs
    # --- Initialize checkpoint trackers ---
    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"./model/best_dc3_model_{case_name}_{epochs}epochs_{timestamp}.pth"

    # 5. Optimization Loop Execution
    print("\nBeginning execution of parallelized training matrix loops for DC3...")
    start_time = time.time()
    for epoch in range(epochs):
        model_dc3.train()
        
        for Pd_batch, Qd_batch in train_loader:
            optimizer_dc3.zero_grad()
            
            # Inner feasibility-repair loop used to construct correction targets
            loss, diag = compute_dc3_qcqp_smax_loss(
                model=model_dc3, 
                Pd_batch=Pd_batch, 
                Qd_batch=Qd_batch, 
                problem=problem, 
                weights=loss_weights_dc3,
                completion_steps=10,
                completion_lr=1e-2,
                corr_steps=5,
                corr_lr=1e-2      
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_dc3.parameters(), 10.0)
            optimizer_dc3.step()
            
        if epoch % 100 == 0:
            # 1. Switch to evaluation mode and freeze gradients
            model_dc3.eval()
            with torch.no_grad():
                # Evaluate the entire validation set at once
                val_loss, val_diag = compute_dc3_qcqp_smax_loss(model_dc3, val_Pd, val_Qd, problem, loss_weights_dc3)

            # 2. Checkpointing Logic: If this is the lowest validation loss we've seen, save it!
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model_dc3._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model_dc3.state_dict()
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