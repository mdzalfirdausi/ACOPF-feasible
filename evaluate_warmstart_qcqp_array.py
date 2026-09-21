#!/usr/bin/env python3
"""
Evaluate NN dispatch predictions as primal initializations for the ORIGINAL
matrix-QCQP Pyomo/IPOPT ACOPF formulation.

Important:
- Loads the CURRENT dataset and preserves its Pd_all/Qd_all.
- Reconstructs M_p, M_q, M_v, M_pf, M_qf, M_pt, M_qt, M_c, M_s in memory
  from the graph/branch coefficients already stored in the current dataset.
- Imports build_acopf_model() from pyomo_ipopt_qcqp.py unchanged.
- Recomputes a cold IPOPT baseline for EVERY selected test instance using the same current formulation and machine.
- Uses NN pg/qg only as the IPOPT primal initialization; voltage remains flat.
- Compares feasible objective gap and runtime against that newly recomputed cold baseline.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import sys
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch

from pyomo_ipopt_qcqp import build_acopf_model
from ACOPF_pinn_baseline import baselineQCQPMLP
from ACOPF_pinn_rahul import RahulSinglePINN_Smax
from ACOPF_Hard_KKT import HardKKT_QCQPMLP


def to_numpy_problem(problem_pt):
    out = {}
    for key, value in problem_pt.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = value
    return out


def reconstruct_qcqp_matrices(problem):
    """
    Reconstruct the exact matrix representation used by the old data generator,
    using the graph coefficients in the CURRENT dataset.
    """
    nbus = int(problem["nbus"])
    nbranch = int(problem["nbranch"])
    D = 2 * nbus

    fbus = np.asarray(problem["fbus"], dtype=int)
    tbus = np.asarray(problem["tbus"], dtype=int)

    g11 = np.asarray(problem["g11"], dtype=float)
    g12 = np.asarray(problem["g12"], dtype=float)
    g21 = np.asarray(problem["g21"], dtype=float)
    g22 = np.asarray(problem["g22"], dtype=float)
    b11 = np.asarray(problem["b11"], dtype=float)
    b12 = np.asarray(problem["b12"], dtype=float)
    b21 = np.asarray(problem["b21"], dtype=float)
    b22 = np.asarray(problem["b22"], dtype=float)

    Gs = np.asarray(problem["Gs"], dtype=float)
    Bs = np.asarray(problem["Bs"], dtype=float)

    M_pf, M_qf, M_pt, M_qt = [], [], [], []

    for l in range(nbranch):
        i = int(fbus[l])
        j = int(tbus[l])
        iB, jB = i + nbus, j + nbus

        A = np.zeros((D, D))
        A[i, i] = g11[l]
        A[iB, iB] = g11[l]
        A[i, j] = -(g12[l] - b21[l])
        A[iB, jB] = -(g12[l] - b21[l])
        A[i, jB] = g21[l] + b12[l]
        A[iB, j] = -(g21[l] + b12[l])
        M_pf.append(0.5 * (A + A.T))

        A = np.zeros((D, D))
        A[i, i] = -b11[l]
        A[iB, iB] = -b11[l]
        A[i, j] = b12[l] + g21[l]
        A[iB, jB] = b12[l] + g21[l]
        A[i, jB] = -(b21[l] - g12[l])
        A[iB, j] = b21[l] - g12[l]
        M_qf.append(0.5 * (A + A.T))

        A = np.zeros((D, D))
        A[j, j] = g22[l]
        A[jB, jB] = g22[l]
        A[j, i] = -(g12[l] + b21[l])
        A[jB, iB] = -(g12[l] + b21[l])
        A[j, iB] = -(g21[l] - b12[l])
        A[jB, i] = g21[l] - b12[l]
        M_pt.append(0.5 * (A + A.T))

        A = np.zeros((D, D))
        A[j, j] = -b22[l]
        A[jB, jB] = -b22[l]
        A[j, i] = b12[l] - g21[l]
        A[jB, iB] = b12[l] - g21[l]
        A[j, iB] = b21[l] + g12[l]
        A[jB, i] = -(b21[l] + g12[l])
        M_qt.append(0.5 * (A + A.T))

    M_p = [np.zeros((D, D)) for _ in range(nbus)]
    M_q = [np.zeros((D, D)) for _ in range(nbus)]

    for i in range(nbus):
        M_p[i][i, i] = Gs[i]
        M_p[i][i + nbus, i + nbus] = Gs[i]
        M_q[i][i, i] = -Bs[i]
        M_q[i][i + nbus, i + nbus] = -Bs[i]

    for l in range(nbranch):
        i, j = int(fbus[l]), int(tbus[l])
        M_p[i] += M_pf[l]
        M_q[i] += M_qf[l]
        M_p[j] += M_pt[l]
        M_q[j] += M_qt[l]

    M_c, M_s = [], []
    for l in range(nbranch):
        i, j = int(fbus[l]), int(tbus[l])
        iB, jB = i + nbus, j + nbus

        A = np.zeros((D, D))
        A[i, j] = 1.0
        A[iB, jB] = 1.0
        M_c.append(0.5 * (A + A.T))

        A = np.zeros((D, D))
        A[iB, j] = 1.0
        A[i, jB] = -1.0
        M_s.append(0.5 * (A + A.T))

    M_v = []
    for i in range(nbus):
        A = np.zeros((D, D))
        A[i, i] = 1.0
        A[i + nbus, i + nbus] = 1.0
        M_v.append(A)

    problem.update({
        "M_p": np.stack(M_p),
        "M_q": np.stack(M_q),
        "M_v": np.stack(M_v),
        "M_pf": np.stack(M_pf),
        "M_qf": np.stack(M_qf),
        "M_pt": np.stack(M_pt),
        "M_qt": np.stack(M_qt),
        "M_c": np.stack(M_c),
        "M_s": np.stack(M_s),
    })
    return problem


def set_loads(m, Pd, Qd):
    for b in m.BUS:
        m.Pd[b] = float(Pd[b])
        m.Qd[b] = float(Qd[b])


def set_cold_start(m, problem, nbus):
    pmin, pmax = problem["pmin"], problem["pmax"]
    qmin, qmax = problem["qmin"], problem["qmax"]
    for g in m.GEN:
        m.pg[g].value = float((pmin[g] + pmax[g]) / 2.0)
        m.qg[g].value = float((qmin[g] + qmax[g]) / 2.0)
    for j in m.BUS2:
        m.v[j].value = 1.0 if j < nbus else 0.0

def set_nn_dispatch_start(m, problem, pg_pred, qg_pred, nbus):
    pmin, pmax = problem["pmin"], problem["pmax"]
    qmin, qmax = problem["qmin"], problem["qmax"]

    for g in m.GEN:
        m.pg[g].value = float(np.clip(pg_pred[g], pmin[g], pmax[g]))
        m.qg[g].value = float(np.clip(qg_pred[g], qmin[g], qmax[g]))

    # Deliberately do not use NN voltage.
    for j in m.BUS2:
        m.v[j].value = 1.0 if j < nbus else 0.0


def success(results):
    return results.solver.termination_condition in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--bus_number", type=int, required=True)
    parser.add_argument("--total_samples", type=int, default=10000)
    parser.add_argument("--eval_limit", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--chunk_id", type=str, default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = f"./dataset/{args.case_name}_{args.total_samples}.pt"
    print(f"Loading CURRENT dataset: {dataset_path}")
    problem_pt = torch.load(dataset_path, map_location="cpu")
    problem_np = reconstruct_qcqp_matrices(to_numpy_problem(problem_pt))
    print("QCQP matrices reconstructed in memory.")

    nbus = int(problem_np["nbus"])
    ngen = int(problem_np["ngen"])
    nbranch = int(problem_np["nbranch"])
    slack_imag_idx = int(np.where(np.asarray(problem_np["a_ref"]) == 1)[0][0])

    n = int(problem_np["Pd_all"].shape[0])
    test_start = int(0.8*n) + int(0.1*n)
    Pd_all = np.asarray(problem_np["Pd_all"][test_start:], dtype=float)
    Qd_all = np.asarray(problem_np["Qd_all"][test_start:], dtype=float)

    # Use the CURRENT test split directly.  The cold baseline is recomputed
    # below with exactly the same formulation, solver, machine, and loads.
    original_ids = np.arange(len(Pd_all), dtype=int)
    Pd = Pd_all
    Qd = Qd_all

    start_idx = max(0, int(args.start_idx))
    end_idx = len(Pd) if args.end_idx is None else min(int(args.end_idx), len(Pd))
    if start_idx >= end_idx:
        raise ValueError(f"Invalid chunk [{start_idx}, {end_idx}) for test size {len(Pd)}")

    Pd = Pd[start_idx:end_idx]
    Qd = Qd[start_idx:end_idx]
    original_ids = original_ids[start_idx:end_idx]

    if args.eval_limit is not None:
        keep = min(int(args.eval_limit), len(Pd))
        Pd, Qd = Pd[:keep], Qd[:keep]
        original_ids = original_ids[:keep]

    n_eval = len(Pd)
    chunk_tag = args.chunk_id if args.chunk_id is not None else f"{start_idx:04d}_{end_idx:04d}"
    print(f"Selected test chunk [{start_idx}, {end_idx}) with {n_eval} instance(s); chunk={chunk_tag}")

    print("\nBuilding ORIGINAL matrix-QCQP Pyomo model...")
    m = build_acopf_model(problem_np, slack_imag_idx)
    solver = pyo.SolverFactory("ipopt")
    solver.options["tol"] = 1e-6
    solver.options["max_iter"] = 3000
    solver.options["max_cpu_time"] = 30.0
    solver.options["warm_start_init_point"] = "no"

    # -----------------------------------------------------------------
    # COLD IPOPT BASELINE: recomputed here for every selected test instance
    # using exactly this matrix-QCQP model and the original cold start.
    # -----------------------------------------------------------------
    print("\n" + "="*68)
    print("COLD IPOPT BASELINE: current formulation, current machine")
    print("="*68)

    cold_times = np.empty(n_eval, dtype=float)
    cold_costs = np.empty(n_eval, dtype=float)
    cold_success = np.zeros(n_eval, dtype=bool)
    cold_status = []

    for i in range(n_eval):
        set_loads(m, Pd[i], Qd[i])
        set_cold_start(m, problem_np, nbus)

        t0 = time.perf_counter()
        try:
            res = solver.solve(m, tee=args.tee)
            cold_times[i] = time.perf_counter() - t0
            ok = success(res)
            term = str(res.solver.termination_condition)
        except Exception as exc:
            cold_times[i] = time.perf_counter() - t0
            ok = False
            term = f"{type(exc).__name__}: {exc}"

        cold_success[i] = ok
        cold_status.append(term)

        if ok:
            pg_sol = np.array([pyo.value(m.pg[g]) for g in m.GEN], dtype=float)
            c2 = np.asarray(problem_np["c2"], dtype=float)
            c1 = np.asarray(problem_np["c1"], dtype=float)
            c0 = np.asarray(problem_np["c0"], dtype=float)
            cold_costs[i] = float(np.sum(c2*pg_sol**2 + c1*pg_sol + c0))
        else:
            cold_costs[i] = np.nan

        if (not ok) or i == 0 or (i + 1) % 25 == 0 or (i + 1) == n_eval:
            print(
                f"  [{i+1:4d}/{n_eval}] success={ok} status={term} "
                f"time={cold_times[i]:.6f}s cost={cold_costs[i]:.6f}"
            )

    # ============================================================
    # Keep only instances with a successful cold-IPOPT reference.
    # The EXACT same subset is then used for every NN architecture/run.
    # ============================================================

    n_requested = n_eval
    valid = cold_success.copy()
    n_valid = int(valid.sum())
    n_failed = int((~valid).sum())

    print(
        f"\nCold IPOPT reference success: "
        f"{n_valid}/{n_requested} "
        f"({100.0 * n_valid / n_requested:.2f}%)"
    )

    if n_failed > 0:
        failed_ids = original_ids[~valid]

        print(f"Cold IPOPT failed on {n_failed} instance(s).")
        print(f"Failed Instance_ID(s): {failed_ids.tolist()}")
        print(
            "These instances are excluded from the paired NN-initialized "
            "IPOPT comparison."
        )

    if n_valid == 0:
        raise RuntimeError(
            "No successful cold-IPOPT reference instances are available."
        )

    # Save the COMPLETE cold-baseline results BEFORE filtering
    cold_df = pd.DataFrame({
        "Instance_ID": original_ids,
        "Cold_IPOPT_Status": cold_status,
        "Cold_IPOPT_Success": cold_success,
        "Cold_IPOPT_Time_s": cold_times,
        "Cold_IPOPT_Cost": cold_costs,
    })

    cold_out = f"result/warmstart_cold_case{args.bus_number}_chunk{chunk_tag}.csv"
    cold_df.to_csv(cold_out, index=False)

    print(f"Cold baseline saved -> {cold_out}")

    # ------------------------------------------------------------
    # Apply ONE common mask to everything used below.
    # ------------------------------------------------------------

    Pd = Pd[valid]
    Qd = Qd[valid]

    original_ids = original_ids[valid]

    cold_times = cold_times[valid]
    cold_costs = cold_costs[valid]

    # From here onward n_eval means the number of valid paired cases.
    n_eval = n_valid

    print(
        f"Proceeding with {n_eval} common reference-feasible "
        f"instances for every architecture and run.\n"
    )

    # Torch problem remains the CURRENT graph representation used by the NNs.
    problem = {}
    for k, v in problem_pt.items():
        if isinstance(v, torch.Tensor):
            problem[k] = v.to(device=device, dtype=torch.float32) if v.is_floating_point() else v.to(device)
        else:
            problem[k] = v

    Pd_t = torch.tensor(Pd, dtype=torch.float32, device=device)
    Qd_t = torch.tensor(Qd, dtype=torch.float32, device=device)

    model_dir = args.model_dir or os.path.join("model", str(args.bus_number))
    def paths(keyword):
        return sorted(glob.glob(os.path.join(model_dir, f"*{keyword}*.pth")))

    archs = {
        "PINN Baseline": (lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device), paths("pinn_model")),
        "DC3": (lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device), paths("dc3_model")),
        "FSNet": (lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device), paths("fsnet_model")),
        "KKT": (lambda: HardKKT_QCQPMLP(nbus, ngen, nbranch, slack_imag_idx).to(device), paths("hardkkt")),
        "Rahul's Model": (lambda: RahulSinglePINN_Smax(nbus, ngen, nbranch).to(device), paths("rahul_model")),
    }

    c2 = np.asarray(problem_np["c2"], dtype=float)
    c1 = np.asarray(problem_np["c1"], dtype=float)
    c0 = np.asarray(problem_np["c0"], dtype=float)

    rows = []

    for arch, (factory, checkpoints) in archs.items():
        print(f"\n================ {arch} ================")
        for run, ckpt in enumerate(checkpoints, 1):
            print(f"Run {run}: {os.path.basename(ckpt)}")
            net = factory()
            sd = torch.load(ckpt, map_location=device, weights_only=True)
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            net.load_state_dict(sd)
            net.float().eval()

            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = net(Pd_t, Qd_t, problem)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                nn_total = time.perf_counter() - t0

            pg_all = out[1].detach().cpu().numpy()
            qg_all = out[2].detach().cpu().numpy()
            nn_per_instance = nn_total / n_eval

            for i in range(n_eval):
                set_loads(m, Pd[i], Qd[i])
                set_nn_dispatch_start(m, problem_np, pg_all[i], qg_all[i], nbus)

                t0 = time.perf_counter()
                try:
                    res = solver.solve(m, tee=args.tee)
                    solve_time = time.perf_counter() - t0
                    ok = success(res)
                    term = str(res.solver.termination_condition)
                except Exception as exc:
                    solve_time = time.perf_counter() - t0
                    ok = False
                    term = f"{type(exc).__name__}: {exc}"

                if ok:
                    pg_sol = np.array([pyo.value(m.pg[g]) for g in m.GEN])
                    cost = float(np.sum(c2*pg_sol**2 + c1*pg_sol + c0))
                    gap = 100.0*(cost-cold_costs[i])/cold_costs[i]
                else:
                    cost, gap = np.nan, np.nan

                total = nn_per_instance + solve_time
                rows.append({
                    "Architecture": arch,
                    "Run": run,
                    "Instance_ID": int(original_ids[i]),
                    "Success": ok,
                    "Status": term,
                    "NN_Time_s": nn_per_instance,
                    "NN_Initialized_IPOPT_Time_s": solve_time,
                    "Total_Online_Time_s": total,
                    "Cold_IPOPT_Time_s": cold_times[i],
                    "Cold_IPOPT_Cost": cold_costs[i],
                    "Speedup_vs_Cold_IPOPT": cold_times[i]/total if total > 0 else np.nan,
                    "Feasible_Cost": cost,
                    "Feasible_Obj_Gap_pct": gap,
                })
                print(f"  [{i+1}/{n_eval}] success={ok} status={term} time={solve_time:.6f}s gap={gap:+.4f}%" if ok
                      else f"  [{i+1}/{n_eval}] success=False status={term}")

            pd.DataFrame(rows).to_csv(
                f"result/warmstart_raw_case{args.bus_number}_chunk{chunk_tag}.csv", index=False
            )

    df = pd.DataFrame(rows)
    if len(df):
        run = df.groupby(["Architecture", "Run"], as_index=False).agg(
            N=("Success", "size"),
            Success_Rate=("Success", "mean"),
            NN_Time_s=("NN_Time_s", "mean"),
            IPOPT_Time_s=("NN_Initialized_IPOPT_Time_s", "mean"),
            Total_Online_Time_s=("Total_Online_Time_s", "mean"),
            Cold_IPOPT_Time_s=("Cold_IPOPT_Time_s", "mean"),
            Speedup=("Speedup_vs_Cold_IPOPT", "mean"),
            Feasible_Obj_Gap_pct=("Feasible_Obj_Gap_pct", "mean"),
        )
        run.to_csv(f"result/warmstart_runlevel_case{args.bus_number}_chunk{chunk_tag}.csv", index=False)

        summary = run.groupby("Architecture").agg(["mean", "std"])
        summary.to_csv(f"result/warmstart_summary_case{args.bus_number}_chunk{chunk_tag}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
