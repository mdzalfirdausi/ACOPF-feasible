#!/usr/bin/env python3
"""
ACOPF Model Evaluation Script
Generates metrics for DC3-style comparison table
"""

import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 1. IMPORT YOUR MODEL CLASSES HERE
from ACOPF_pinn_baseline import baselineQCQPMLP
from ACOPF_pinn_rahul import RahulSinglePINN_Smax

# --- UTILS ---
def quad_batch_stack(v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,kij,bj->bk", v, M, v)

# --- CORE EVALUATION FUNCTION ---
def evaluate_model(model: nn.Module, model_name: str, test_loader: DataLoader, problem: dict, device: torch.device):
    model.eval()
    
    total_samples = 0
    total_time = 0.0
    
    all_objs = []
    all_max_eq, all_mean_eq = [], []
    all_max_ineq, all_mean_ineq = [], []
    
    # Pre-extract bounds to avoid redundant batch expansions
    smax = problem["smax"].unsqueeze(0)
    angmax = problem["angmax"].unsqueeze(0)
    angmin = problem["angmin"].unsqueeze(0)
    Vmin = problem["Vmin"].unsqueeze(0)
    Vmax = problem["Vmax"].unsqueeze(0)
    pmax = problem["pmax"].unsqueeze(0)
    pmin = problem["pmin"].unsqueeze(0)
    qmax = problem["qmax"].unsqueeze(0)
    qmin = problem["qmin"].unsqueeze(0)
    c2, c1, c0 = problem["c2"].unsqueeze(0), problem["c1"].unsqueeze(0), problem["c0"].unsqueeze(0)
    
    with torch.no_grad():
        for Pd_batch, Qd_batch in test_loader:
            B = Pd_batch.shape[0]
            total_samples += B
            
            # --- Timing Inference ---
            start_time = time.perf_counter()
            
            # Handle different forward signatures
            if model_name == "Rahul Model":
                # Rahul model returns 15 variables
                outputs = model(Pd_batch, Qd_batch)
                v, pg, qg = outputs[0], outputs[1], outputs[2]
            else:
                # Baseline, DC3, FSNet expect problem dict and return 3 variables
                v, pg, qg = model(Pd_batch, Qd_batch, problem)
                
            end_time = time.perf_counter()
            total_time += (end_time - start_time)

            # --- Objective Value ---
            cost_per_gen = c2.expand(B,-1) * (pg ** 2) + c1.expand(B,-1) * pg + c0.expand(B,-1)
            obj = cost_per_gen.sum(dim=1)
            all_objs.extend(obj.cpu().numpy())

            # --- Evaluate Quadratic Forms ---
            vp = quad_batch_stack(v, problem["M_p"])
            vq = quad_batch_stack(v, problem["M_q"])
            pf = quad_batch_stack(v, problem["M_pf"])
            qf = quad_batch_stack(v, problem["M_qf"])
            pt = quad_batch_stack(v, problem["M_pt"])
            qt = quad_batch_stack(v, problem["M_qt"])
            vc = quad_batch_stack(v, problem["M_c"])
            vs = quad_batch_stack(v, problem["M_s"])
            vv = quad_batch_stack(v, problem["M_v"])

            # --- Equality Constraints (Power Balance) ---
            h_p = (pg @ problem["C_g"].T) - Pd_batch - vp
            h_q = (qg @ problem["C_g"].T) - Qd_batch - vq
            
            eq_violations = torch.cat([h_p.abs(), h_q.abs()], dim=1)
            all_max_eq.append(eq_violations.max().item())
            all_mean_eq.append(eq_violations.mean().item())

            # --- Inequality Constraints ---
            g_sf = (pf**2 + qf**2) - smax.expand(B,-1)**2
            g_st = (pt**2 + qt**2) - smax.expand(B,-1)**2
            g_pg_max = pg - pmax.expand(B,-1)
            g_pg_min = pmin.expand(B,-1) - pg
            g_qg_max = qg - qmax.expand(B,-1)
            g_qg_min = qmin.expand(B,-1) - qg
            g_ang_min = torch.tan(angmin.expand(B,-1)) * vc - vs
            g_ang_max = vs - torch.tan(angmax.expand(B,-1)) * vc
            g_v_max = vv - (Vmax.expand(B,-1)**2)
            g_v_min = (Vmin.expand(B,-1)**2) - vv

            ineq_violations = torch.cat([
                F.relu(g_sf), F.relu(g_st), 
                F.relu(g_pg_max), F.relu(g_pg_min), F.relu(g_qg_max), F.relu(g_qg_min),
                F.relu(g_ang_min), F.relu(g_ang_max), 
                F.relu(g_v_max), F.relu(g_v_min)
            ], dim=1)
            
            all_max_ineq.append(ineq_violations.max().item())
            all_mean_ineq.append(ineq_violations.mean().item())

    # --- Aggregate Metrics ---
    metrics = {
        "Obj. Value": f"{np.mean(all_objs):.2f} ({np.std(all_objs):.2f})",
        "Max Eq.": f"{max(all_max_eq):.4f} ({np.std(all_max_eq):.4f})",
        "Mean Eq.": f"{np.mean(all_mean_eq):.4f} ({np.std(all_mean_eq):.4f})",
        "Max Ineq.": f"{max(all_max_ineq):.4f} ({np.std(all_max_ineq):.4f})",
        "Mean Ineq.": f"{np.mean(all_mean_ineq):.4f} ({np.std(all_mean_ineq):.4f})",
        "Time (s)": f"{(total_time / total_samples):.6f} (0.0000)"
    }
    return metrics

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load Data
    case_name = 'pglib_opf_case14_ieee'
    total_samples = 10000
    dataset_path = f'./dataset/{case_name}_{total_samples}.pt'
    problem = torch.load(dataset_path, map_location=device)

    # 2. Extract EXACTLY the Test Set (The remaining 10%)
    actual_total_samples = problem["Pd_all"].shape[0] 
    train_size = int(0.8 * actual_total_samples)
    val_size = int(0.1 * actual_total_samples)
    test_start = train_size + val_size

    test_Pd = problem["Pd_all"][test_start:].to(device)
    test_Qd = problem["Qd_all"][test_start:].to(device)

    # Deploy matrices to device
    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            problem[key] = value.to(device)

    batch_size = 1024 
    test_dataset = TensorDataset(test_Pd, test_Qd)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()
    nbus = problem["nbus"]
    ngen = problem["ngen"]
    nbranch = problem["nbranch"]

    # 3. Model Registry - Map your files to your classes here!
    models_to_evaluate = {
        "PINN Baseline": {
            "path": "./model/pinn_model_pglib_opf_case14_ieee_10000epochs.pth",
            "class": baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device)
        },
        "DC3 Model": {
            "path": "./model/dc3_model_pglib_opf_case14_ieee_10000epochs.pth",
            "class": baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device)
        },
        "FSNet Model": {
            "path": "./model/fsnet_model_pglib_opf_case14_ieee_10000epochs.pth",
            "class": baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device)
        },
        "Rahul Model": {
            "path": "./model/rahul_pinn_pglib_opf_case14_ieee_10000epochs.pth",
            "class": RahulSinglePINN_Smax(nbus, ngen, nbranch).to(device)
        }
    }

    # 4. Evaluation Loop
    results = {}
    print("\n" + "="*85)
    print(f"{'Method':<15} | {'Obj. Value':<12} | {'Max Eq.':<12} | {'Mean Eq.':<12} | {'Max Ineq.':<12} | {'Time (s)'}")
    print("-" * 85)

    for model_name, config in models_to_evaluate.items():
        model = config["class"]
        model.load_state_dict(torch.load(config["path"], map_location=device, weights_only=True))
        
        metrics = evaluate_model(model, model_name, test_loader, problem, device)
        results[model_name] = metrics
        
        # Print row
        print(f"{model_name:<15} | {metrics['Obj. Value']:<12} | {metrics['Max Eq.']:<12} | {metrics['Mean Eq.']:<12} | {metrics['Max Ineq.']:<12} | {metrics['Time (s)']}")
    
    print("="*85 + "\n")