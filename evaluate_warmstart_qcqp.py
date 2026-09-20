#!/usr/bin/env python3
"""
Evaluate NN dispatch predictions as primal initializations for the ORIGINAL
matrix-QCQP Pyomo/IPOPT ACOPF formulation.

Important:
- Loads the CURRENT dataset and preserves its Pd_all/Qd_all.
- Reconstructs M_p, M_q, M_v, M_pf, M_qf, M_pt, M_qt, M_c, M_s in memory
  from the graph/branch coefficients already stored in the current dataset.
- Imports build_acopf_model() from pyomo_ipopt_qcqp.py unchanged.
- Performs a cold-start sanity check BEFORE evaluating any NN.
- Uses NN pg/qg only as the IPOPT primal initialization; voltage remains flat.
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
    parser.add_argument("--eval_limit", type=int, default=10)
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

    gt_path = f"./result/ipopt_baseline_{args.case_name}_{n-test_start}_instances.npz"
    gt = np.load(gt_path)
    status = gt["status"]
    mask = np.array([("ok" in str(s).lower()) or ("optimal" in str(s).lower()) for s in status])
    original_ids = np.where(mask)[0]

    Pd = Pd_all[mask]
    Qd = Qd_all[mask]
    gt_pg = np.asarray(gt["pg_optimal"][mask], dtype=float)
    gt_time = np.asarray(gt["solve_time"][mask], dtype=float)

    n_eval = min(args.eval_limit, len(Pd))
    Pd, Qd = Pd[:n_eval], Qd[:n_eval]
    gt_pg, gt_time = gt_pg[:n_eval], gt_time[:n_eval]
    original_ids = original_ids[:n_eval]

    print("\nBuilding ORIGINAL matrix-QCQP Pyomo model...")
    m = build_acopf_model(problem_np, slack_imag_idx)
    solver = pyo.SolverFactory("ipopt")
    solver.options["tol"] = 1e-6
    solver.options["max_iter"] = 3000
    solver.options["warm_start_init_point"] = "no"

    # -----------------------------------------------------------------
    # MANDATORY SANITY CHECK
    # -----------------------------------------------------------------
    print("\n" + "="*68)
    print("SANITY CHECK: original matrix-QCQP model with original cold start")
    print("="*68)
    set_loads(m, Pd[0], Qd[0])
    set_cold_start(m, problem_np, nbus)

    t0 = time.perf_counter()
    sanity = solver.solve(m, tee=args.tee)
    sanity_time = time.perf_counter() - t0
    sanity_ok = success(sanity)
    print(f"termination = {sanity.solver.termination_condition}")
    print(f"success     = {sanity_ok}")
    print(f"solve time  = {sanity_time:.6f} s")
    print(f"stored time = {gt_time[0]:.6f} s")

    if not sanity_ok:
        print("\nSTOP: the reconstructed matrix-QCQP model did not reproduce a")
        print("successful cold IPOPT solve. No NN warm-start experiment was run.")
        sys.exit(2)

    print("\nSANITY CHECK PASSED. Proceeding to NN dispatch initialization.\n")

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

    c2, c1, c0 = problem_np["c2"], problem_np["c1"], problem_np["c0"]
    gt_cost = np.sum(c2[None, :]*gt_pg**2 + c1[None, :]*gt_pg + c0[None, :], axis=1)

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
                    gap = 100.0*(cost-gt_cost[i])/gt_cost[i]
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
                    "Stored_Cold_IPOPT_Time_s": gt_time[i],
                    "Speedup_vs_Stored_Cold": gt_time[i]/total if total > 0 else np.nan,
                    "Feasible_Cost": cost,
                    "Feasible_Obj_Gap_pct": gap,
                })
                print(f"  [{i+1}/{n_eval}] success={ok} status={term} time={solve_time:.6f}s gap={gap:+.4f}%" if ok
                      else f"  [{i+1}/{n_eval}] success=False status={term}")

            pd.DataFrame(rows).to_csv(
                f"result/warmstart_raw_case{args.bus_number}.csv", index=False
            )

    df = pd.DataFrame(rows)
    if len(df):
        run = df.groupby(["Architecture", "Run"], as_index=False).agg(
            N=("Success", "size"),
            Success_Rate=("Success", "mean"),
            NN_Time_s=("NN_Time_s", "mean"),
            IPOPT_Time_s=("NN_Initialized_IPOPT_Time_s", "mean"),
            Total_Online_Time_s=("Total_Online_Time_s", "mean"),
            Cold_IPOPT_Time_s=("Stored_Cold_IPOPT_Time_s", "mean"),
            Speedup=("Speedup_vs_Stored_Cold", "mean"),
            Feasible_Obj_Gap_pct=("Feasible_Obj_Gap_pct", "mean"),
        )
        run.to_csv(f"result/warmstart_runlevel_case{args.bus_number}.csv", index=False)

        summary = run.groupby("Architecture").agg(["mean", "std"])
        summary.to_csv(f"result/warmstart_summary_case{args.bus_number}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
