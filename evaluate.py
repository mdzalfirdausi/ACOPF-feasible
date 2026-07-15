#!/usr/bin/env python3
"""
ACOPF Model Evaluation Script
Generates metrics for DC3-style comparison table and rigorous performance plots.
"""

import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    all_mae_v, all_mae_pg, all_mae_qg = [], [], []
    
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
    
    # NEW: Store system-level metrics for plotting
    plot_data = {
        "ipopt_costs": [], 
        "nn_costs": [], 
        "max_violations": []
    }

    with torch.no_grad():
        for Pd_batch, Qd_batch, v_gt, pg_gt, qg_gt in test_loader:
            # Force all inputs and ground truths to float32 to prevent any mismatch
            Pd_batch, Qd_batch = Pd_batch.float(), Qd_batch.float()
            v_gt, pg_gt, qg_gt = v_gt.float(), pg_gt.float(), qg_gt.float()
            
            B = Pd_batch.shape[0]
            total_samples += B
            
            # --- Timing Inference ---
            start_time = time.perf_counter()
            
            # Handle different forward signatures
            if model_name == "Rahul Model":
                outputs = model(Pd_batch, Qd_batch)
                v, pg, qg = outputs[0], outputs[1], outputs[2]
            else:
                v, pg, qg = model(Pd_batch, Qd_batch, problem)
                
            total_time += (time.perf_counter() - start_time)

            # Distance from Ground Truth (MAE)
            all_mae_v.append(torch.abs(v - v_gt).mean().item())
            all_mae_pg.append(torch.abs(pg - pg_gt).mean().item())
            all_mae_qg.append(torch.abs(qg - qg_gt).mean().item())

            # --- Objective Value (NN and IPOPT) ---
            cost_nn = c2.expand(B,-1) * (pg ** 2) + c1.expand(B,-1) * pg + c0.expand(B,-1)
            cost_ipopt = c2.expand(B,-1) * (pg_gt ** 2) + c1.expand(B,-1) * pg_gt + c0.expand(B,-1)
            
            obj = cost_nn.sum(dim=1)
            all_objs.extend(obj.cpu().numpy())
            
            # Store for plotting
            plot_data["nn_costs"].extend(cost_nn.sum(dim=1).cpu().numpy())
            plot_data["ipopt_costs"].extend(cost_ipopt.sum(dim=1).cpu().numpy())

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
            
            # Combine Eq and Ineq to find the absolute worst violation per sample for plotting
            batch_max_eq = eq_violations.max(dim=1).values
            batch_max_ineq = ineq_violations.max(dim=1).values
            batch_max_viol = torch.max(batch_max_eq, batch_max_ineq)
            plot_data["max_violations"].extend(batch_max_viol.cpu().numpy())

    # --- Aggregate Metrics for Table ---
    metrics = {
        "Obj. Value": f"{np.mean(all_objs):.2f}",
        "Max Eq.": f"{max(all_max_eq):.4f}",
        "Mean Eq.": f"{np.mean(all_mean_eq):.4f}",
        "Max Ineq.": f"{max(all_max_ineq):.4f}",
        "MAE v": np.mean(all_mae_v),
        "MAE pg": np.mean(all_mae_pg),
        "MAE qg": np.mean(all_mae_qg),
        "Time (s)": total_time / total_samples
    }
    
    # Convert plot lists to numpy arrays
    plot_data["nn_costs"] = np.array(plot_data["nn_costs"])
    plot_data["ipopt_costs"] = np.array(plot_data["ipopt_costs"])
    plot_data["max_violations"] = np.array(plot_data["max_violations"])
    
    return metrics, plot_data

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

    test_Pd = problem["Pd_all"][test_start:].to(device, dtype=torch.float32)
    test_Qd = problem["Qd_all"][test_start:].to(device, dtype=torch.float32)

    # Load IPOPT Ground Truth for the Test Set
    gt_path = f'./result/ipopt_baseline_{case_name}_{actual_total_samples - test_start}_instances.npz'
    try:
        gt_data = np.load(gt_path)
    except FileNotFoundError:
        print(f"CRITICAL: Ground truth file not found at {gt_path}. Required for gap plotting.")
        sys.exit(1)

    # Filter out failed IPOPT solves
    status = gt_data['status']
    mask = np.array(['ok' in s.lower() or 'optimal' in s.lower() for s in status])
    print(f"Total Test Instances: {len(mask)} | Successful IPOPT Solves: {mask.sum()}")

    # Extract ground truth variables (only for successful solves)
    test_v_gt = torch.tensor(gt_data['v_optimal'][mask], dtype=torch.float32).to(device)
    test_pg_gt = torch.tensor(gt_data['pg_optimal'][mask], dtype=torch.float32).to(device)
    test_qg_gt = torch.tensor(gt_data['qg_optimal'][mask], dtype=torch.float32).to(device)
    
    # Apply mask to NN inputs to ensure alignment
    test_Pd = test_Pd[mask]
    test_Qd = test_Qd[mask]

    assert test_Pd.shape[0] == test_v_gt.shape[0], "Dataset size mismatch between PINN test set and IPOPT baseline!"

    # Deploy matrices to device
    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            # Cast floating-point tensors to float32, leave integer tensors (like indices) alone
            if value.is_floating_point():
                problem[key] = value.to(device, dtype=torch.float32)
            else:
                problem[key] = value.to(device)

    # Build DataLoader with 5 variables
    batch_size = 1024 
    test_dataset = TensorDataset(test_Pd, test_Qd, test_v_gt, test_pg_gt, test_qg_gt)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()
    nbus = problem["nbus"]
    ngen = problem["ngen"]
    nbranch = problem["nbranch"]

    # 3. Model Registry
    models_to_evaluate = {
        "PINN Baseline": {
            "path": "./model/best_pinn_model_pglib_opf_case14_ieee_10000epochs_20260702_104818.pth",
            "class": baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device)
        },
        "DC3 Model": {
            "path": "./model/best_dc3_model_pglib_opf_case14_ieee_10000epochs_20260702_112123.pth",
            "class": baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device)
        },
        "FSNet Model": {
            "path": "./model/best_fsnet_model_pglib_opf_case14_ieee_10000epochs_20260702_115656.pth",
            "class": baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device)
        },
        "Rahul Model": {
            "path": "./model/rahul_pinn_pglib_opf_case14_ieee_10000epochs.pth",
            "class": RahulSinglePINN_Smax(nbus, ngen, nbranch).to(device)
        }
    }

    # 4. Evaluation Loop
    results_list = []
    plot_artifacts = {}

    for model_name, config in models_to_evaluate.items():
        model = config["class"]
        try:
            model.load_state_dict(torch.load(config["path"], map_location=device, weights_only=True))
            model = model.to(device).float() 
            metrics, plot_data = evaluate_model(model, model_name, test_loader, problem, device)
            metrics["Model"] = model_name
            results_list.append(metrics)
            plot_artifacts[model_name] = plot_data
            
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")

    # Display as Pandas DataFrame
    df_results = pd.DataFrame(results_list)
    if not df_results.empty:
        df_results = df_results[["Model", "Obj. Value", "Max Eq.", "Mean Eq.", "Max Ineq.", "MAE v", "MAE pg", "MAE qg", "Time (s)"]]
    else:
        print("WARNING: No models were successfully evaluated. DataFrame is empty.")
    print("\n--- MODEL PERFORMANCE METRICS ---")
    try:
        from IPython.display import display
        display(df_results)
    except ImportError:
        print(df_results.to_string())

    # =========================================================================
    # PLOTTING SECTION
    # =========================================================================
    print("\nGenerating rigorous validation plots...")

    # -------------------------------------------------------------
    # PLOT 2: Max Physical Violations (Sorted Error Curve - 2x2 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, model_name in enumerate(models_to_evaluate.keys()):
        if model_name in plot_artifacts:
            ax = axes[i]
            viols = plot_artifacts[model_name]["max_violations"]
            
            # Sort the violations to create a clean curve
            sorted_viols = np.sort(viols)
            
            # Plot the sorted violations
            ax.scatter(range(len(sorted_viols)), sorted_viols, alpha=0.6, s=15, color='red')
            
            # Formatting (Log Scale is crucial here)
            ax.set_yscale('log')
            ax.axhline(y=1e-4, color='k', linestyle='--', linewidth=2, label='Tolerance (1e-4)')
            
            ax.set_title(f"{model_name}: Physical Feasibility")
            ax.set_xlabel("Sample Index (Sorted by Error)")
            ax.set_ylabel("Max Constraint Violation (p.u.) [Log Scale]")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("plot/sorted_error_curves.pdf", format="pdf", bbox_inches="tight")
    plt.show()  

    # -------------------------------------------------------------
    # PLOT 3: Distribution of Maximum Physical Violations (Boxplot)
    # -------------------------------------------------------------
    model_names = []
    all_model_viols = []

    for model_name in models_to_evaluate.keys():
        if model_name in plot_artifacts:
            model_names.append(model_name)
            viols = plot_artifacts[model_name]["max_violations"]
            viols_safe = np.clip(viols, a_min=1e-10, a_max=None) 
            all_model_viols.append(viols_safe)

    if all_model_viols:
        plt.figure(figsize=(10, 6))
        
        box = plt.boxplot(all_model_viols, patch_artist=True)
        # Safely apply labels regardless of matplotlib version
        plt.xticks(ticks=range(1, len(model_names) + 1), labels=model_names)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        plt.yscale('log')
        plt.axhline(y=1e-4, color='r', linestyle='--', linewidth=2, label='Acceptable Solver Tolerance (1e-4)')
        
        plt.title("Distribution of Maximum Physical Violations Across Models", fontsize=14)
        plt.ylabel("Max Constraint Violation (p.u.) [Log Scale]", fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='upper left')
        
        plt.tight_layout()
        plt.savefig("plot/violation_boxplots.pdf", format="pdf", bbox_inches="tight")
        plt.show()
    else:
        print("No violation data available to plot.")