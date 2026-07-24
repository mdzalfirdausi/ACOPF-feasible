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
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 1. IMPORT YOUR MODEL CLASSES HERE
# (Ensure whatever class KKT uses is imported here if different from baselineQCQPMLP)
from ACOPF_pinn_baseline import baselineQCQPMLP
from ACOPF_pinn_rahul import RahulSinglePINN_Smax
from ACOPF_Hard_KKT import HardKKT_QCQPMLP

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
    
    # Pre-extract Grid Graph Topology Tensors
    nbus = problem["nbus"]
    fbus = problem["fbus"]
    tbus = problem["tbus"]
    
    g11, g12 = problem["g11"].unsqueeze(0), problem["g12"].unsqueeze(0)
    g21, g22 = problem["g21"].unsqueeze(0), problem["g22"].unsqueeze(0)
    b11, b12 = problem["b11"].unsqueeze(0), problem["b12"].unsqueeze(0)
    b21, b22 = problem["b21"].unsqueeze(0), problem["b22"].unsqueeze(0)
    
    Gs, Bs = problem["Gs"].unsqueeze(0), problem["Bs"].unsqueeze(0)
    
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
            
            # Handle different forward signatures and return lengths
            if "Rahul" in model_name:
                outputs = model(Pd_batch, Qd_batch, problem)
            else:
                outputs = model(Pd_batch, Qd_batch, problem)
                
            # Safely extract the 3 primal variables (v, pg, qg) whether the model 
            # returns just 3 items or 15 items (like Hard KKT duals)
            v, pg, qg = outputs[0], outputs[1], outputs[2]
            
            total_time += (time.perf_counter() - start_time)

            # Distance from Ground Truth (MAE)
            all_mae_v.append(torch.abs(v - v_gt).mean().item())
            all_mae_pg.append(torch.abs(pg - pg_gt).mean().item())
            all_mae_qg.append(torch.abs(qg - qg_gt).mean().item())

            # --- Objective Value Error vs IPOPT (%) ---
            cost_nn = c2.expand(B,-1) * (pg ** 2) + c1.expand(B,-1) * pg + c0.expand(B,-1)
            cost_ipopt = c2.expand(B,-1) * (pg_gt ** 2) + c1.expand(B,-1) * pg_gt + c0.expand(B,-1)
            
            obj_nn = cost_nn.sum(dim=1)
            obj_ipopt = cost_ipopt.sum(dim=1)
            
            # Calculate Signed Relative Percentage Error (Optimality Gap)
            obj_gap_pct = ((obj_nn - obj_ipopt) / obj_ipopt) * 100.0
            
            # Store signed gap
            all_objs.extend(obj_gap_pct.cpu().numpy())
            plot_data["nn_costs"].extend(obj_nn.cpu().numpy())
            plot_data["ipopt_costs"].extend(obj_ipopt.cpu().numpy())

            # --- Evaluate Physics using Sparse Graph Constraints ---
            vr = v[:, :nbus]
            vi = v[:, nbus:]
            vv = vr**2 + vi**2  # Nodal voltage squared [B, nbus]
            
            # Slice voltages to branch from/to endpoints
            vr_f, vi_f = vr[:, fbus], vi[:, fbus]
            vr_t, vi_t = vr[:, tbus], vi[:, tbus]
            
            vv_f = vr_f**2 + vi_f**2
            vv_t = vr_t**2 + vi_t**2
            
            # Voltage angle differences (cos/sin representations)
            vc = vr_f * vr_t + vi_f * vi_t
            vs = vi_f * vr_t - vr_f * vi_t
            
            # Branch Active & Reactive Power Flows
            pf = g11 * vv_f + g12 * vc + b12 * vs
            qf = -b11 * vv_f - b12 * vc + g12 * vs
            
            pt = g22 * vv_t + g21 * vc - b21 * vs
            qt = -b22 * vv_t - b21 * vc - g21 * vs
            
            # Nodal Power Injections (Starts with Shunt Consumption)
            vp = Gs.expand(B, -1) * vv
            vq = -Bs.expand(B, -1) * vv
            
            # Aggregate Branch Flows into Nodes using scatter_add
            fbus_exp = fbus.unsqueeze(0).expand(B, -1)
            tbus_exp = tbus.unsqueeze(0).expand(B, -1)
            
            vp.scatter_add_(1, fbus_exp, pf)
            vp.scatter_add_(1, tbus_exp, pt)
            
            vq.scatter_add_(1, fbus_exp, qf)
            vq.scatter_add_(1, tbus_exp, qt)

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
    # === Setup Command Line Arguments ===
    parser = argparse.ArgumentParser(description="Evaluate Architectures.")
    parser.add_argument('--case_name', type=str, required=True, help="Name of the grid case (e.g., pglib_opf_case3_lmbd)")
    parser.add_argument('--bus_number', type=int, required=True, help="Number of buses in the grid case")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # =========================================================================
    # SYSTEM CONFIGURATION
    # =========================================================================
    bus_number = args.bus_number  # <--- ONLY CHANGE THIS NUMBER (e.g., 14, 57, 162, 300)
    # Note: Adjust the string format below if case 162 has a different suffix like '_ieee_dtc'
    case_name = args.case_name  # <--- ONLY CHANGE THIS STRING (e.g., pglib_opf_case3_lmbd, pglib_opf_case14_ieee, etc.)
    total_samples = 10000
    model_dir = os.path.join(r"M:\projects\ACOPF-feasible\model", str(bus_number))
    
    print(f"Targeting Dataset: {case_name}")
    print(f"Looking for models in: {model_dir}")
    # =========================================================================

    # 1. Load Data
    dataset_path = f'./dataset/{case_name}_{total_samples}.pt'
    try:
        problem = torch.load(dataset_path, map_location=device)
    except FileNotFoundError:
        print(f"CRITICAL: Dataset not found at {dataset_path}")
        sys.exit(1)

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

    # 3. Dynamic Multi-Seed Model Registry
    def get_model_paths(arch_keyword):
        """Helper to find all matching model runs alphabetically (by timestamp)"""
        search_pattern = os.path.join(model_dir, f"*{arch_keyword}*.pth")
        paths = sorted(glob.glob(search_pattern))
        if not paths:
            print(f" ⚠️ WARNING: No checkpoints found matching '{arch_keyword}' in {model_dir}")
        return paths

    architectures_config = {
        "PINN Baseline": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": get_model_paths("pinn_model")
        },
        "DC3": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": get_model_paths("dc3_model")
        },
        "FSNet": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": get_model_paths("fsnet_model")
        },
        "KKT": {
            "class": lambda: HardKKT_QCQPMLP(nbus, ngen, nbranch, slack_imag_idx).to(device),
            "paths": get_model_paths("hardkkt")
        },
        "Rahul's Model": {
            "class": lambda: RahulSinglePINN_Smax(nbus, ngen, nbranch).to(device),
            "paths": get_model_paths("rahul_model")
        }
    }

    # 4. Evaluation Loop
    raw_results_list = []
    # Structure: arch_plot_artifacts[arch_name] = [plot_data_run1, plot_data_run2, ...]
    arch_plot_artifacts = {arch: [] for arch in architectures_config.keys()}

    for arch_name, config in architectures_config.items():
        print(f"\n--- Evaluating Architecture: {arch_name} ---")
        
        if not config["paths"]:
            print(f"  [Skipped] No checkpoints available.")
            continue
            
        for run_idx, path in enumerate(config["paths"]):
            model = config["class"]()
            try:
                # Inside evaluate_new.py around line 300:
                state_dict = torch.load(path, map_location=device, weights_only=True)
                # Strip '_orig_mod.' prefix added by torch.compile
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
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
        print("TABLE 1: INDIVIDUAL CHECKPOINT EVALUATIONS (ALL IDENTIFIED RUNS)")
        print("=========================================================================================")
        df_display_raw = df_raw.copy()
        df_display_raw["Obj. Error (%)"] = df_display_raw.apply(lambda r: f"{r['Obj_Mean']:.2f} ({r['Obj_Std']:.2f})", axis=1)
        df_display_raw = df_display_raw[["Architecture", "Run", "Obj. Error (%)", "Max_Eq", "Mean_Eq", "Max_Ineq", "Mean_Ineq", "MAE_v", "MAE_pg", "MAE_qg", "Time_s"]]
        print(df_display_raw.to_string(index=False))

        print("\n=========================================================================================")
        print("TABLE 2: PAPER-READY COMPARISON TABLE (AGGREGATED MEAN ± STD ACROSS SEEDS)")
        print("=========================================================================================")
        
        summary_rows = []
        for arch_name, group in df_raw.groupby("Architecture", sort=False):
            n_seeds = len(group)
            mean_gap = group['Obj_Mean'].mean()
            std_gap = group['Obj_Mean'].std()
            
            # Format explicitly with sign (+/-) so reviewers see cost undercutting
            gap_str = f"{mean_gap:+.4f} ± {std_gap:.4f}"
            
            summary_rows.append({
                "Architecture": f"{arch_name} (n={n_seeds})",
                "Optimality Gap (%)": gap_str,  
                "Max Eq. (p.u.)": f"{group['Max_Eq'].mean():.4f} ± {group['Max_Eq'].std():.4f}",
                "Mean Eq. (p.u.)": f"{group['Mean_Eq'].mean():.4f} ± {group['Mean_Eq'].std():.4f}",
                "Max Ineq. (p.u.)": f"{group['Max_Ineq'].mean():.4f} ± {group['Max_Ineq'].std():.4f}",
                "Mean Ineq. (p.u.)": f"{group['Mean_Ineq'].mean():.4f} ± {group['Mean_Ineq'].std():.4f}",
                "MAE v": f"{group['MAE_v'].mean():.5f}",
                "MAE pg": f"{group['MAE_pg'].mean():.4f}",
                "MAE qg": f"{group['MAE_qg'].mean():.4f}",
                "Time (s)": f"{group['Time_s'].mean():.6f} ± {group['Time_s'].std():.6f}"
            })
            
        df_summary = pd.DataFrame(summary_rows)
        try:
            from IPython.display import display
            print(df_summary.to_string(index=False))
        except ImportError:
            print(df_summary.to_string(index=False))
            
        os.makedirs("plot", exist_ok=True)
        df_summary.to_csv(f"plot/paper_comparison_table_case{bus_number}.csv", index=False)
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
            
            for r_idx, run_artifact in enumerate(runs_data):
                viols = run_artifact["max_violations"]
                sorted_viols = np.sort(viols)
                all_sorted_viols.append(sorted_viols)
                
                ax.plot(range(len(sorted_viols)), sorted_viols, alpha=0.25, color='red', 
                        linewidth=1, label='Individual Runs' if r_idx == 0 else "")
            
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

    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(f"plot/sorted_error_curves_multiseed_case{bus_number}.pdf", format="pdf", bbox_inches="tight")

    # -------------------------------------------------------------
    # PLOT 2: Pooled Distribution of Maximum Violations
    # -------------------------------------------------------------
    plot_rows = []
    for arch_name, runs_data in arch_plot_artifacts.items():
        if runs_data:
            combined_viols = np.concatenate([r["max_violations"] for r in runs_data])
            viols_safe = np.clip(combined_viols, a_min=1e-10, a_max=None)
            
            log_viols = np.log10(viols_safe)
            arch_label = f"{arch_name}"
            
            for lv in log_viols:
                plot_rows.append({
                    "Architecture": arch_label,
                    "Log_Max_Violation": lv
                })

    if plot_rows:
        df_plot = pd.DataFrame(plot_rows)
        
        plt.figure(figsize=(12, 6.5))
        
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        ax = sns.violinplot(
            data=df_plot,
            x="Architecture",
            y="Log_Max_Violation",
            density_norm="count",  
            inner="box",           
            cut=0,                 
            palette=palette[:df_plot["Architecture"].nunique()],
            linewidth=1.2,
            alpha=0.75
        )
        
        plt.xticks(fontsize=10.5, fontweight='bold')
        
        plt.axhline(y=-4.0, color='r', linestyle='--', linewidth=2, label='Solver Tolerance ($10^{-4}$)')
        
        y_min = int(np.floor(df_plot["Log_Max_Violation"].min()))
        y_max = int(np.ceil(df_plot["Log_Max_Violation"].max()))
        tick_locs = np.arange(y_min, y_max + 1, 1)
        plt.yticks(tick_locs, [f"$10^{{{int(loc)}}}$" for loc in tick_locs], fontsize=11)
        
        plt.title(f"Constraint Violation Distribution Across {len(mask)} Test Instances (Case {bus_number})", fontsize=14, fontweight='bold')
        plt.xlabel("", fontsize=12) 
        plt.ylabel("Max Constraint Violation (p.u.)", fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=11, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"plot/violation_violinplots_pooled_case{bus_number}.pdf", format="pdf", bbox_inches="tight")
    else:
        print("No violation data available to plot.")