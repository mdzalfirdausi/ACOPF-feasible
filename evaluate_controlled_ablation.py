#!/usr/bin/env python3

import os
import re
import glob
import time
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ACOPF_controlled_ablation import baselineQCQPMLP


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

CASES = {
    14: "pglib_opf_case14_ieee",    
    57: "pglib_opf_case57_ieee",
    162: "pglib_opf_case162_ieee_dtc",
    300: "pglib_opf_case300_ieee",
}

PROJECTION_CONFIGS = [
    ("none", "nominal"),
    ("voltage", "nominal"),
    ("generation", "nominal"),
    ("full", "nominal"),
]

WEIGHT_CONFIGS = [
    ("full", "equal"),
    ("full", "inequality"),
    ("full", "physics"),
    ("full", "objective"),
]

ALL_CONFIGS = PROJECTION_CONFIGS + WEIGHT_CONFIGS

SEEDS = [1, 2, 3, 4, 5]

FEAS_TOL = 1e-4


# ============================================================
# PATH SELECTION
# ============================================================

def checkpoint_pattern(case_name, projection, weights, seed):
    return (
        f"best_controlled_{projection}_{weights}_"
        f"{case_name}_10000epochs_seed{seed}_*.pth"
    )


def select_checkpoint(model_root, bus, case_name, projection, weights, seed):
    """
    Select exactly one checkpoint.

    162/300:
        Prefer the authoritative ISS archive.

    14/57:
        Search recursively and choose the newest timestamped checkpoint.
    """

    pattern = checkpoint_pattern(
        case_name, projection, weights, seed
    )

    if bus == 162:
        authoritative = (
            Path(model_root)
            / "controlled_ablation_20260922"
            / "iss_4090"
            / "case162"
        )

        matches = sorted(authoritative.glob(pattern))

    elif bus == 300:
        authoritative = (
            Path(model_root)
            / "controlled_ablation_20260922"
            / "iss_4090"
            / "case300"
        )

        matches = sorted(authoritative.glob(pattern))

    else:
        matches = sorted(
            Path(model_root).rglob(pattern)
        )

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found:\n"
            f"bus={bus}, projection={projection}, "
            f"weights={weights}, seed={seed}"
        )

    # Filenames end in YYYYMMDD_HHMMSS.pth.
    # Lexicographic ordering therefore selects latest timestamp.
    selected = matches[-1]

    return selected, matches


# ============================================================
# DATA
# ============================================================

def load_case(case_name, device):

    dataset_path = (
        f"./dataset/{case_name}_10000.pt"
    )

    problem = torch.load(
        dataset_path,
        map_location=device,
    )

    total = problem["Pd_all"].shape[0]

    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    test_start = train_size + val_size

    test_Pd = problem["Pd_all"][test_start:].to(
        device, dtype=torch.float32
    )

    test_Qd = problem["Qd_all"][test_start:].to(
        device, dtype=torch.float32
    )

    gt_path = (
        f"./result/ipopt_baseline_"
        f"{case_name}_{total-test_start}_instances.npz"
    )

    gt = np.load(gt_path)

    status = gt["status"]

    mask = np.array([
        ("ok" in str(s).lower())
        or ("optimal" in str(s).lower())
        for s in status
    ])

    test_Pd = test_Pd[mask]
    test_Qd = test_Qd[mask]

    v_gt = torch.tensor(
        gt["v_optimal"][mask],
        dtype=torch.float32,
        device=device,
    )

    pg_gt = torch.tensor(
        gt["pg_optimal"][mask],
        dtype=torch.float32,
        device=device,
    )

    qg_gt = torch.tensor(
        gt["qg_optimal"][mask],
        dtype=torch.float32,
        device=device,
    )

    for key, value in problem.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                problem[key] = value.to(
                    device,
                    dtype=torch.float32,
                )
            else:
                problem[key] = value.to(device)

    dataset = TensorDataset(
        test_Pd,
        test_Qd,
        v_gt,
        pg_gt,
        qg_gt,
    )

    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
    )

    return problem, loader, mask


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_checkpoint(
    model,
    loader,
    problem,
    device,
):

    model.eval()

    rows = []

    total_time = 0.0
    total_samples = 0
    instance_id = 0

    smax = problem["smax"].unsqueeze(0)
    angmax = problem["angmax"].unsqueeze(0)
    angmin = problem["angmin"].unsqueeze(0)

    Vmin = problem["Vmin"].unsqueeze(0)
    Vmax = problem["Vmax"].unsqueeze(0)

    pmax = problem["pmax"].unsqueeze(0)
    pmin = problem["pmin"].unsqueeze(0)

    qmax = problem["qmax"].unsqueeze(0)
    qmin = problem["qmin"].unsqueeze(0)

    c2 = problem["c2"].unsqueeze(0)
    c1 = problem["c1"].unsqueeze(0)
    c0 = problem["c0"].unsqueeze(0)

    nbus = problem["nbus"]

    fbus = problem["fbus"]
    tbus = problem["tbus"]

    g11 = problem["g11"].unsqueeze(0)
    g12 = problem["g12"].unsqueeze(0)
    g21 = problem["g21"].unsqueeze(0)
    g22 = problem["g22"].unsqueeze(0)

    b11 = problem["b11"].unsqueeze(0)
    b12 = problem["b12"].unsqueeze(0)
    b21 = problem["b21"].unsqueeze(0)
    b22 = problem["b22"].unsqueeze(0)

    Gs = problem["Gs"].unsqueeze(0)
    Bs = problem["Bs"].unsqueeze(0)

    with torch.no_grad():

        for (
            Pd,
            Qd,
            v_gt,
            pg_gt,
            qg_gt,
        ) in loader:

            B = Pd.shape[0]

            # -----------------------------
            # inference timing
            # -----------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            outputs = model(
                Pd,
                Qd,
                problem,
            )

            v, pg, qg = (
                outputs[0],
                outputs[1],
                outputs[2],
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_samples += B

            # -----------------------------
            # objective
            # -----------------------------

            cost_nn = (
                c2.expand(B, -1) * pg**2
                + c1.expand(B, -1) * pg
                + c0.expand(B, -1)
            )

            cost_ipopt = (
                c2.expand(B, -1) * pg_gt**2
                + c1.expand(B, -1) * pg_gt
                + c0.expand(B, -1)
            )

            obj_nn = cost_nn.sum(dim=1)
            obj_ipopt = cost_ipopt.sum(dim=1)

            obj_gap = (
                (obj_nn - obj_ipopt)
                / obj_ipopt
            ) * 100.0

            # -----------------------------
            # voltages
            # -----------------------------

            vr = v[:, :nbus]
            vi = v[:, nbus:]

            vv = vr**2 + vi**2

            vr_f = vr[:, fbus]
            vi_f = vi[:, fbus]

            vr_t = vr[:, tbus]
            vi_t = vi[:, tbus]

            vv_f = vr_f**2 + vi_f**2
            vv_t = vr_t**2 + vi_t**2

            v_rt_cross = (
                vr_f * vr_t
                + vi_f * vi_t
            )

            v_it_cross = (
                vr_f * vi_t
                - vi_f * vr_t
            )

            # -----------------------------
            # branch flows
            # -----------------------------

            pf = (
                g11 * vv_f
                - (g12 - b21) * v_rt_cross
                + (g21 + b12) * v_it_cross
            )

            qf = (
                -b11 * vv_f
                + (b12 + g21) * v_rt_cross
                + (b21 - g12) * v_it_cross
            )

            pt = (
                g22 * vv_t
                - (g12 + b21) * v_rt_cross
                + (g21 - b12) * v_it_cross
            )

            qt = (
                -b22 * vv_t
                + (b12 - g21) * v_rt_cross
                - (b21 + g12) * v_it_cross
            )

            # -----------------------------
            # nodal injections
            # -----------------------------

            vp = Gs.expand(B, -1) * vv
            vq = -Bs.expand(B, -1) * vv

            fbus_exp = (
                fbus.unsqueeze(0)
                .expand(B, -1)
            )

            tbus_exp = (
                tbus.unsqueeze(0)
                .expand(B, -1)
            )

            vp.scatter_add_(
                1,
                fbus_exp,
                pf,
            )

            vp.scatter_add_(
                1,
                tbus_exp,
                pt,
            )

            vq.scatter_add_(
                1,
                fbus_exp,
                qf,
            )

            vq.scatter_add_(
                1,
                tbus_exp,
                qt,
            )

            # -----------------------------
            # equality violations
            # -----------------------------

            h_p = (
                pg @ problem["C_g"].T
                - Pd
                - vp
            )

            h_q = (
                qg @ problem["C_g"].T
                - Qd
                - vq
            )

            eq = torch.cat(
                [h_p.abs(), h_q.abs()],
                dim=1,
            )

            max_eq = eq.max(dim=1).values
            mean_eq = eq.mean(dim=1)

            # -----------------------------
            # inequalities
            # -----------------------------

            g_sf = (
                pf**2 + qf**2
                - smax.expand(B, -1)**2
            )

            g_st = (
                pt**2 + qt**2
                - smax.expand(B, -1)**2
            )

            g_pg_max = (
                pg - pmax.expand(B, -1)
            )

            g_pg_min = (
                pmin.expand(B, -1) - pg
            )

            g_qg_max = (
                qg - qmax.expand(B, -1)
            )

            g_qg_min = (
                qmin.expand(B, -1) - qg
            )

            g_ang_min = (
                torch.tan(
                    angmin.expand(B, -1)
                )
                * v_rt_cross
                - v_it_cross
            )

            g_ang_max = (
                v_it_cross
                - torch.tan(
                    angmax.expand(B, -1)
                )
                * v_rt_cross
            )

            g_v_max = (
                vv
                - Vmax.expand(B, -1)**2
            )

            g_v_min = (
                Vmin.expand(B, -1)**2
                - vv
            )

            ineq = torch.cat(
                [
                    F.relu(g_sf),
                    F.relu(g_st),
                    F.relu(g_pg_max),
                    F.relu(g_pg_min),
                    F.relu(g_qg_max),
                    F.relu(g_qg_min),
                    F.relu(g_ang_min),
                    F.relu(g_ang_max),
                    F.relu(g_v_max),
                    F.relu(g_v_min),
                ],
                dim=1,
            )

            max_ineq = (
                ineq.max(dim=1).values
            )

            mean_ineq = (
                ineq.mean(dim=1)
            )

            max_violation = torch.maximum(
                max_eq,
                max_ineq,
            )

            feasible = (
                max_violation <= FEAS_TOL
            ).float()

            # -----------------------------
            # instance rows
            # -----------------------------

            for j in range(B):

                rows.append({
                    "Instance_ID":
                        instance_id + j,

                    "Objective_Deviation_pct":
                        obj_gap[j].item(),

                    "Max_Eq":
                        max_eq[j].item(),

                    "Mean_Eq":
                        mean_eq[j].item(),

                    "Max_Ineq":
                        max_ineq[j].item(),

                    "Mean_Ineq":
                        mean_ineq[j].item(),

                    "Max_Violation":
                        max_violation[j].item(),

                    "Feasible_1e-4":
                        feasible[j].item(),
                })

            instance_id += B

    df = pd.DataFrame(rows)

    metrics = {
        "Objective_Deviation_pct":
            df["Objective_Deviation_pct"].mean(),

        "Max_Eq":
            df["Max_Eq"].mean(),

        "Mean_Eq":
            df["Mean_Eq"].mean(),

        "Max_Ineq":
            df["Max_Ineq"].mean(),

        "Mean_Ineq":
            df["Mean_Ineq"].mean(),

        "Max_Violation":
            df["Max_Violation"].mean(),

        "Feasibility_Rate":
            df["Feasible_1e-4"].mean(),

        "Inference_Time_s":
            total_time / total_samples,
    }

    return metrics, df


# ============================================================
# HIERARCHICAL BOOTSTRAP
# ============================================================

def hierarchical_bootstrap(
    df,
    metric,
    n_boot=10000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    runs = sorted(df["Seed"].unique())

    boot = np.empty(n_boot)

    for b in range(n_boot):

        sampled_runs = rng.choice(
            runs,
            size=len(runs),
            replace=True,
        )

        run_values = []

        for run in sampled_runs:

            values = (
                df.loc[
                    df["Seed"] == run,
                    metric,
                ]
                .to_numpy()
            )

            sampled_values = rng.choice(
                values,
                size=len(values),
                replace=True,
            )

            run_values.append(
                sampled_values.mean()
            )

        boot[b] = np.mean(run_values)

    return (
        np.percentile(boot, 2.5),
        np.percentile(boot, 97.5),
    )


# ============================================================
# SUMMARY
# ============================================================

METRICS = [
    "Objective_Deviation_pct",
    "Max_Eq",
    "Mean_Eq",
    "Max_Ineq",
    "Mean_Ineq",
    "Max_Violation",
    "Feasibility_Rate",
    "Inference_Time_s",
]


def create_summary(
    run_df,
    instance_df,
):

    rows = []

    group_cols = [
        "Bus",
        "Case",
        "Projection",
        "Weights",
    ]

    for keys, runs in run_df.groupby(group_cols):

        bus, case, projection, weights = keys

        inst = instance_df[
            (instance_df["Bus"] == bus)
            & (instance_df["Projection"] == projection)
            & (instance_df["Weights"] == weights)
        ]

        row = {
            "Bus": bus,
            "Case": case,
            "Projection": projection,
            "Weights": weights,
            "N_Runs": len(runs),
        }

        for metric in METRICS:

            values = runs[metric].to_numpy()

            row[f"{metric}_Mean"] = (
                values.mean()
            )

            row[f"{metric}_SD"] = (
                values.std(ddof=1)
            )

            # inference time has no meaningful
            # per-instance nested observations here
            if metric != "Inference_Time_s":

                lo, hi = hierarchical_bootstrap(
                    inst,
                    metric
                    if metric != "Feasibility_Rate"
                    else "Feasible_1e-4",
                )

                row[f"{metric}_CI_L"] = lo
                row[f"{metric}_CI_U"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# RUN-LEVEL PERMUTATION TEST
# ============================================================

def paired_permutation_pvalue(
    x,
    y,
):

    x = np.asarray(x)
    y = np.asarray(y)

    d = x - y

    observed = abs(d.mean())

    # Exact sign-flip test:
    # only 2^5 = 32 possibilities
    statistics = []

    for mask in range(2 ** len(d)):

        signs = np.array([
            1 if (mask >> i) & 1
            else -1
            for i in range(len(d))
        ])

        statistics.append(
            abs((d * signs).mean())
        )

    statistics = np.asarray(statistics)

    return np.mean(
        statistics >= observed - 1e-15
    )


def pairwise_tests(
    run_df,
    experiment,
):

    rows = []

    for bus in sorted(
        run_df["Bus"].unique()
    ):

        case_df = run_df[
            run_df["Bus"] == bus
        ].copy()

        if experiment == "projection":

            case_df = case_df[
                case_df["Weights"] == "nominal"
            ].copy()

            case_df["Config"] = (
                case_df["Projection"]
            )

        elif experiment == "weighting":

            case_df = case_df[
                case_df["Projection"] == "full"
            ].copy()

            case_df["Config"] = (
                case_df["Weights"]
            )

        else:
            raise ValueError(experiment)

        configs = sorted(
            case_df["Config"].unique()
        )

        for metric in METRICS:

            if metric == "Inference_Time_s":
                # still valid at run level
                pass

            metric_rows = []

            for a, b in combinations(
                configs,
                2,
            ):

                da = (
                    case_df[
                        case_df["Config"] == a
                    ]
                    .sort_values("Seed")
                )

                db = (
                    case_df[
                        case_df["Config"] == b
                    ]
                    .sort_values("Seed")
                )

                merged = da[
                    ["Seed", metric]
                ].merge(
                    db[["Seed", metric]],
                    on="Seed",
                    suffixes=("_A", "_B"),
                )

                if len(merged) != 5:
                    raise RuntimeError(
                        f"{bus} {a} vs {b}: "
                        f"expected 5 paired seeds"
                    )

                x = merged[
                    f"{metric}_A"
                ].to_numpy()

                y = merged[
                    f"{metric}_B"
                ].to_numpy()

                p = paired_permutation_pvalue(
                    x,
                    y,
                )

                metric_rows.append({
                    "Bus": bus,
                    "Experiment": experiment,
                    "Metric": metric,
                    "Config_A": a,
                    "Config_B": b,
                    "Mean_A": x.mean(),
                    "Mean_B": y.mean(),
                    "Mean_Difference_A_minus_B":
                        (x - y).mean(),
                    "P_raw": p,
                })

            if metric_rows:

                pvals = [
                    r["P_raw"]
                    for r in metric_rows
                ]

                _, p_adj, _, _ = (
                    multipletests(
                        pvals,
                        method="holm",
                    )
                )

                for r, adj in zip(
                    metric_rows,
                    p_adj,
                ):
                    r["P_Holm"] = adj
                    rows.append(r)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_root",
        default="model",
    )

    parser.add_argument(
        "--output_dir",
        default="result/controlled_ablation",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    manifest_rows = []
    run_rows = []
    instance_frames = []

    for bus, case_name in CASES.items():

        print()
        print("=" * 70)
        print(f"CASE {bus}: {case_name}")
        print("=" * 70)

        problem, loader, mask = load_case(
            case_name,
            device,
        )

        nbus = problem["nbus"]
        ngen = problem["ngen"]

        slack_imag_idx = (
            (problem["a_ref"] == 1)
            .nonzero(as_tuple=True)[0]
            .item()
        )

        print(
            f"Successful IPOPT test instances: "
            f"{mask.sum()} / {len(mask)}"
        )

        for projection, weights in ALL_CONFIGS:

            for seed in SEEDS:

                path, all_matches = (
                    select_checkpoint(
                        args.model_root,
                        bus,
                        case_name,
                        projection,
                        weights,
                        seed,
                    )
                )

                print(
                    f"{bus} | {projection:10s} | "
                    f"{weights:10s} | "
                    f"seed={seed} | "
                    f"{path}"
                )

                manifest_rows.append({
                    "Bus": bus,
                    "Case": case_name,
                    "Projection": projection,
                    "Weights": weights,
                    "Seed": seed,
                    "Selected_Checkpoint":
                        str(path),
                    "Candidate_Count":
                        len(all_matches),
                })

                model = baselineQCQPMLP(
                    nbus=nbus,
                    ngen=ngen,
                    slack_imag_idx=slack_imag_idx,
                    projection_mode=projection,
                ).to(device)

                state = torch.load(
                    path,
                    map_location=device,
                    weights_only=True,
                )

                state = {
                    k.replace(
                        "_orig_mod.",
                        "",
                    ): v
                    for k, v in state.items()
                }

                model.load_state_dict(state)

                metrics, df_inst = (
                    evaluate_checkpoint(
                        model,
                        loader,
                        problem,
                        device,
                    )
                )

                run_row = {
                    "Bus": bus,
                    "Case": case_name,
                    "Projection": projection,
                    "Weights": weights,
                    "Seed": seed,
                    "Checkpoint": str(path),
                    **metrics,
                }

                run_rows.append(run_row)

                df_inst["Bus"] = bus
                df_inst["Case"] = case_name
                df_inst["Projection"] = projection
                df_inst["Weights"] = weights
                df_inst["Seed"] = seed

                instance_frames.append(
                    df_inst
                )

    manifest = pd.DataFrame(
        manifest_rows
    )

    run_df = pd.DataFrame(
        run_rows
    )

    instance_df = pd.concat(
        instance_frames,
        ignore_index=True,
    )

    manifest.to_csv(
        os.path.join(
            args.output_dir,
            "checkpoint_manifest.csv",
        ),
        index=False,
    )

    run_df.to_csv(
        os.path.join(
            args.output_dir,
            "run_level.csv",
        ),
        index=False,
    )

    instance_df.to_csv(
        os.path.join(
            args.output_dir,
            "instance_level.csv",
        ),
        index=False,
    )

    summary = create_summary(
        run_df,
        instance_df,
    )

    summary.to_csv(
        os.path.join(
            args.output_dir,
            "summary.csv",
        ),
        index=False,
    )

    projection_summary = summary[
        summary["Weights"] == "nominal"
    ].copy()

    projection_summary.to_csv(
        os.path.join(
            args.output_dir,
            "projection_ablation_summary.csv",
        ),
        index=False,
    )

    weighting_summary = summary[
        summary["Projection"] == "full"
    ].copy()

    weighting_summary.to_csv(
        os.path.join(
            args.output_dir,
            "weighting_ablation_summary.csv",
        ),
        index=False,
    )

    projection_tests = pairwise_tests(
        run_df,
        "projection",
    )

    projection_tests.to_csv(
        os.path.join(
            args.output_dir,
            "projection_pairwise_tests.csv",
        ),
        index=False,
    )

    weighting_tests = pairwise_tests(
        run_df,
        "weighting",
    )

    weighting_tests.to_csv(
        os.path.join(
            args.output_dir,
            "weighting_pairwise_tests.csv",
        ),
        index=False,
    )

    print()
    print("=" * 70)
    print("CONTROLLED ABLATION EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "Models evaluated:",
        len(run_df),
    )

    print(
        "Instance-level rows:",
        len(instance_df),
    )

    print(
        "Output:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
