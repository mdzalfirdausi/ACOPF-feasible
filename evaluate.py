#!/usr/bin/env python3
"""
ACOPF Multi-Seed Model Evaluation Script
Generates aggregated metrics (Mean ± Std across seeds) for DC3-style comparison tables 
and rigorous pooled performance plots across 5 architectures x 5 runs each.
"""
import os
# Prevent OpenMP runtime crash on Windows Conda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
# (Ensure whatever class KKT uses is imported here if different from baselineQCQPMLP)
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
    
    plot_data = {
        "ipopt_costs": [], 
        "nn_costs": [], 
        "max_violations": []
    }

    with torch.no_grad():
        for Pd_batch, Qd_batch, v_gt, pg_gt, qg_gt in test_loader:
            Pd_batch, Qd_batch = Pd_batch.float(), Qd_batch.float()
            v_gt, pg_gt, qg_gt = v_gt.float(), pg_gt.float(), qg_gt.float()
            
            B = Pd_batch.shape[0]
            total_samples += B
            
            # --- Timing Inference ---
            start_time = time.perf_counter()
            
            # Handle different forward signatures
            if "Rahul" in model_name:
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
            
            # Absolute worst violation per sample for plotting
            batch_max_eq = eq_violations.max(dim=1).values
            batch_max_ineq = ineq_violations.max(dim=1).values
            batch_max_viol = torch.max(batch_max_eq, batch_max_ineq)
            plot_data["max_violations"].extend(batch_max_viol.cpu().numpy())

    # Return raw numerical means for aggregation across seeds
    raw_metrics = {
        "Obj_Mean": np.mean(all_objs),
        "Obj_Std": np.std(all_objs),
        "Max_Eq": np.max(all_max_eq),
        "Mean_Eq": np.mean(all_mean_eq),
        "Max_Ineq": np.max(all_max_ineq),
        "Mean_Ineq": np.mean(all_mean_ineq),
        "MAE_v": np.mean(all_mae_v),
        "MAE_pg": np.mean(all_mae_pg),
        "MAE_qg": np.mean(all_mae_qg),
        "Time_s": total_time / total_samples
    }
    
    plot_data["nn_costs"] = np.array(plot_data["nn_costs"])
    plot_data["ipopt_costs"] = np.array(plot_data["ipopt_costs"])
    plot_data["max_violations"] = np.array(plot_data["max_violations"])
    
    return raw_metrics, plot_data

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

    status = gt_data['status']
    mask = np.array(['ok' in s.lower() or 'optimal' in s.lower() for s in status])
    print(f"Total Test Instances: {len(mask)} | Successful IPOPT Solves: {mask.sum()}")

    test_v_gt = torch.tensor(gt_data['v_optimal'][mask], dtype=torch.float32).to(device)
    test_pg_gt = torch.tensor(gt_data['pg_optimal'][mask], dtype=torch.float32).to(device)
    test_qg_gt = torch.tensor(gt_data['qg_optimal'][mask], dtype=torch.float32).to(device)
    
    test_Pd = test_Pd[mask]
    test_Qd = test_Qd[mask]

    assert test_Pd.shape[0] == test_v_gt.shape[0], "Dataset size mismatch between PINN test set and IPOPT baseline!"

    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                problem[key] = value.to(device, dtype=torch.float32)
            else:
                problem[key] = value.to(device)

    batch_size = 1024 
    test_dataset = TensorDataset(test_Pd, test_Qd, test_v_gt, test_pg_gt, test_qg_gt)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    slack_imag_idx = (problem["a_ref"] == 1).nonzero(as_tuple=True)[0].item()
    nbus = problem["nbus"]
    ngen = problem["ngen"]
    nbranch = problem["nbranch"]

    # 3. Multi-Seed Model Registry (5 Architectures x 5 Runs)
    # TODO: Fill in the remaining checkpoint paths for PINN Baseline, FSNet, KKT, and Rahul.
    architectures_config = {
        "DC3": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": [
                "./model/best_dc3_model_pglib_opf_case14_ieee_10000epochs_20260715_202917.pth",
                "./model/best_dc3_model_pglib_opf_case14_ieee_10000epochs_20260715_204723.pth",
                "./model/best_dc3_model_pglib_opf_case14_ieee_10000epochs_20260715_210541.pth",
                "./model/best_dc3_model_pglib_opf_case14_ieee_10000epochs_20260716_155933.pth",
                "./model/best_dc3_model_pglib_opf_case14_ieee_10000epochs_20260716_161731.pth",
            ]
        },
        "PINN Baseline": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": [
                "./model/best_pinn_model_pglib_opf_case14_ieee_10000epochs_20260702_104818.pth",
                # ADD 4 MORE PATHS HERE
            ]
        },
        "FSNet": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": [
                "./model/best_fsnet_model_pglib_opf_case14_ieee_10000epochs_20260702_115656.pth",
                # ADD 4 MORE PATHS HERE
            ]
        },
        "KKT": {
            # Assuming KKT shares the baseline QCQP MLP structure. Change if using a different class.
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": [
                # ADD 5 PATHS HERE
            ]
        },
        "Rahul Model": {
            "class": lambda: RahulSinglePINN_Smax(nbus, ngen, nbranch).to(device),
            "paths": [
                "./model/rahul_pinn_pglib_opf_case14_ieee_10000epochs.pth",
                # ADD 4 MORE PATHS HERE
            ]
        }
    }

    # 4. Evaluation Loop
    raw_results_list = []
    # Structure: arch_plot_artifacts[arch_name] = [plot_data_run1, plot_data_run2, ...]
    arch_plot_artifacts = {arch: [] for arch in architectures_config.keys()}

    for arch_name, config in architectures_config.items():
        print(f"\n--- Evaluating Architecture: {arch_name} ---")
        for run_idx, path in enumerate(config["paths"]):
            if not os.path.exists(path):
                print(f"  [Run {run_idx+1}] Skipped: Path not found -> {path}")
                continue
                
            model = config["class"]()
            try:
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model = model.to(device).float() 
                
                raw_metrics, plot_data = evaluate_model(model, arch_name, test_loader, problem, device)
                
                # Store identifying metadata
                raw_metrics["Architecture"] = arch_name
                raw_metrics["Run"] = f"Run {run_idx + 1}"
                raw_results_list.append(raw_metrics)
                
                arch_plot_artifacts[arch_name].append(plot_data)
                print(f"  [Run {run_idx+1}] Evaluated successfully | Obj: {raw_metrics['Obj_Mean']:.2f} | Max Viol: {raw_metrics['Max_Ineq']:.4f}")
                
            except Exception as e:
                print(f"  [Run {run_idx+1}] Failed due to error: {e}")

    # =========================================================================
    # METRICS AGGREGATION & REPORTING
    # =========================================================================
    df_raw = pd.DataFrame(raw_results_list)
    
    if not df_raw.empty:
        print("\n=========================================================================================")
        print("TABLE 1: INDIVIDUAL CHECKPOINT EVALUATIONS (ALL 25 RUNS)")
        print("=========================================================================================")
        df_display_raw = df_raw.copy()
        df_display_raw["Obj. Value"] = df_display_raw.apply(lambda r: f"{r['Obj_Mean']:.2f} ({r['Obj_Std']:.2f})", axis=1)
        df_display_raw = df_display_raw[["Architecture", "Run", "Obj. Value", "Max_Eq", "Mean_Eq", "Max_Ineq", "Mean_Ineq", "MAE_v", "MAE_pg", "MAE_qg", "Time_s"]]
        print(df_display_raw.to_string(index=False))

        print("\n=========================================================================================")
        print("TABLE 2: PAPER-READY COMPARISON TABLE (AGGREGATED MEAN ± STD ACROSS SEEDS)")
        print("=========================================================================================")
        
        # Calculate Mean and Std across the seeds for each architecture
        summary_rows = []
        for arch_name, group in df_raw.groupby("Architecture", sort=False):
            n_seeds = len(group)
            summary_rows.append({
                "Architecture": f"{arch_name} (n={n_seeds})",
                "Obj. Value": f"{group['Obj_Mean'].mean():.2f} ± {group['Obj_Mean'].std():.2f}",
                "Max Eq. (p.u.)": f"{group['Max_Eq'].mean():.4f} ± {group['Max_Eq'].std():.4f}",
                "Mean Eq. (p.u.)": f"{group['Mean_Eq'].mean():.4f} ± {group['Mean_Eq'].std():.4f}",
                "Max Ineq. (p.u.)": f"{group['Max_Ineq'].mean():.4f} ± {group['Max_Ineq'].std():.4f}",
                "Mean Ineq. (p.u.)": f"{group['Mean_Ineq'].mean():.4f} ± {group['Mean_Ineq'].std():.4f}",
                "MAE v": f"{group['MAE_v'].mean():.5f}",
                "MAE pg": f"{group['MAE_pg'].mean():.4f}",
                "MAE qg": f"{group['MAE_qg'].mean():.4f}",
                "Time (s)": f"{group['Time_s'].mean():.6f}"
            })
            
        df_summary = pd.DataFrame(summary_rows)
        try:
            from IPython.display import display
            display(df_summary)
        except ImportError:
            print(df_summary.to_string(index=False))
            
        # Optional: Save tables to CSV for LaTeX importing
        os.makedirs("plot", exist_ok=True)
        df_summary.to_csv("plot/paper_comparison_table.csv", index=False)
    else:
        print("WARNING: No models were successfully evaluated.")
        sys.exit(0)

    # =========================================================================
    # PLOTTING SECTION
    # =========================================================================
    print("\nGenerating rigorous validation plots across all seeds...")
    os.makedirs("plot", exist_ok=True)

    # -------------------------------------------------------------
    # PLOT 1: Sorted Error Curves (2x3 Grid to fit 5 Architectures)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, (arch_name, runs_data) in enumerate(arch_plot_artifacts.items()):
        ax = axes[i]
        if runs_data:
            all_sorted_viols = []
            
            # Plot individual seeds as light background curves
            for r_idx, run_artifact in enumerate(runs_data):
                viols = run_artifact["max_violations"]
                sorted_viols = np.sort(viols)
                all_sorted_viols.append(sorted_viols)
                
                # Thin transparent line for individual seeds
                ax.plot(range(len(sorted_viols)), sorted_viols, alpha=0.25, color='red', 
                        linewidth=1, label='Individual Runs' if r_idx == 0 else "")
            
            # Compute and plot Median curve across the seeds
            min_len = min(len(v) for v in all_sorted_viols)
            stacked_viols = np.vstack([v[:min_len] for v in all_sorted_viols])
            median_curve = np.median(stacked_viols, axis=0)
            
            ax.plot(range(min_len), median_curve, color='darkred', linewidth=2.5, label='Median across seeds')
            
            ax.set_yscale('log')
            ax.axhline(y=1e-4, color='k', linestyle='--', linewidth=1.5, label='Tolerance (1e-4)')
            
            ax.set_title(f"{arch_name} ({len(runs_data)} seeds)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Sample Index (Sorted by Error)")
            ax.set_ylabel("Max Violation (p.u.) [Log Scale]")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left', fontsize=9)
        else:
            ax.set_title(f"{arch_name} (No Data)")
            ax.axis('off')

    # Hide the 6th empty subplot in the 2x3 grid
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig("plot/sorted_error_curves_multiseed.pdf", format="pdf", bbox_inches="tight")
    # plt.show()

    # -------------------------------------------------------------
    # PLOT 2: Pooled Distribution of Maximum Violations (Boxplot)
    # -------------------------------------------------------------
    model_names = []
    pooled_model_viols = []

    for arch_name, runs_data in arch_plot_artifacts.items():
        if runs_data:
            model_names.append(arch_name)
            # Pool all test sample violations across all 5 seeds (e.g., 1000 samples * 5 runs = 5000 points)
            combined_viols = np.concatenate([r["max_violations"] for r in runs_data])
            viols_safe = np.clip(combined_viols, a_min=1e-10, a_max=None) 
            pooled_model_viols.append(viols_safe)

    if pooled_model_viols:
        plt.figure(figsize=(11, 6))
        
        box = plt.boxplot(pooled_model_viols, patch_artist=True)
        plt.xticks(ticks=range(1, len(model_names) + 1), labels=model_names, fontsize=11, fontweight='bold')
        
        # Color palette for up to 5 architectures
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
        for patch, color in zip(box['boxes'], colors[:len(pooled_model_viols)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        plt.yscale('log')
        plt.axhline(y=1e-4, color='r', linestyle='--', linewidth=2, label='Acceptable Solver Tolerance (1e-4)')
        
        plt.title("Pooled Physical Feasibility Distribution Across All Seeds", fontsize=14, fontweight='bold')
        plt.ylabel("Max Constraint Violation (p.u.) [Log Scale]", fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=11, loc='upper left')
        
        plt.tight_layout()
        plt.savefig("plot/violation_boxplots_pooled.pdf", format="pdf", bbox_inches="tight")
        # plt.show()
    else:
        print("No violation data available to plot.")