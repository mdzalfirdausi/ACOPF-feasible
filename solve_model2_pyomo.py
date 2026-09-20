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
    parser = argparse.ArgumentParser(description="Solve Model 2 QCQP ACOPF with Pyomo/IPOPT.")
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--total_samples", type=int, default=10000)
    parser.add_argument("--eval_limit", type=int, default=1000)
    parser.add_argument("--instance", type=int, default=None,
                        help="Solve one displayed test instance (1-based), e.g. 46.")
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    dataset_path = f"./dataset/{args.case_name}_{args.total_samples}.pt"
    print(f"Loading CURRENT dataset: {dataset_path}")
    problem_pt = torch.load(dataset_path, map_location="cpu")
    problem_np = reconstruct_qcqp_matrices(to_numpy_problem(problem_pt))
    print("QCQP matrices reconstructed in memory.")

    nbus = int(problem_np["nbus"])
    ngen = int(problem_np["ngen"])
    slack_imag_idx = int(np.where(np.asarray(problem_np["a_ref"]) == 1)[0][0])

    n = int(problem_np["Pd_all"].shape[0])
    test_start = int(0.8*n) + int(0.1*n)
    Pd = np.asarray(problem_np["Pd_all"][test_start:], dtype=float)
    Qd = np.asarray(problem_np["Qd_all"][test_start:], dtype=float)

    if args.instance is not None:
        idx = args.instance - 1
        if idx < 0 or idx >= len(Pd):
            raise ValueError(f"--instance must be 1..{len(Pd)}")
        indices = [idx]
    else:
        indices = list(range(min(args.eval_limit, len(Pd))))

    print("\nBuilding Model 2 in Pyomo...")
    m = build_acopf_model(problem_np, slack_imag_idx)
    solver = pyo.SolverFactory("ipopt")
    solver.options["tol"] = 1e-6
    solver.options["max_iter"] = 3000
    solver.options["warm_start_init_point"] = "no"

    rows, Vs, PGs, QGs = [], [], [], []

    for k, i in enumerate(indices, 1):
        set_loads(m, Pd[i], Qd[i])
        set_cold_start(m, problem_np, nbus)

        t0 = time.perf_counter()
        try:
            res = solver.solve(m, tee=args.tee)
            elapsed = time.perf_counter() - t0
            ok = success(res)
            term = str(res.solver.termination_condition)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            ok = False
            term = f"{type(exc).__name__}: {exc}"

        if ok:
            v = np.array([pyo.value(m.v[j]) for j in m.BUS2], dtype=float)
            pg = np.array([pyo.value(m.pg[g]) for g in m.GEN], dtype=float)
            qg = np.array([pyo.value(m.qg[g]) for g in m.GEN], dtype=float)
            obj = float(pyo.value(m.cost))

            max_eq = max_ineq = 0.0
            for con in m.component_data_objects(pyo.Constraint, active=True):
                body = pyo.value(con.body)
                if con.equality:
                    max_eq = max(max_eq, abs(body - pyo.value(con.lower)))
                else:
                    viol = 0.0
                    if con.has_lb():
                        viol = max(viol, pyo.value(con.lower) - body)
                    if con.has_ub():
                        viol = max(viol, body - pyo.value(con.upper))
                    max_ineq = max(max_ineq, max(0.0, viol))
        else:
            v = np.full(2*nbus, np.nan)
            pg = np.full(ngen, np.nan)
            qg = np.full(ngen, np.nan)
            obj = max_eq = max_ineq = np.nan

        Vs.append(v); PGs.append(pg); QGs.append(qg)
        rows.append({
            "Instance": i+1, "Success": ok, "Status": term,
            "Objective": obj, "Solve_Time_s": elapsed,
            "Max_Eq_Violation": max_eq, "Max_Ineq_Violation": max_ineq,
            "Total_Pd": float(Pd[i].sum()), "Total_Qd": float(Qd[i].sum())
        })

        if ok:
            print(f"[{k}/{len(indices)}] instance={i+1} optimal "
                  f"time={elapsed:.6f}s obj={obj:.6f} "
                  f"max_eq={max_eq:.3e} max_ineq={max_ineq:.3e}")
        else:
            print(f"[{k}/{len(indices)}] instance={i+1} FAILED status={term} "
                  f"time={elapsed:.6f}s")

    df = pd.DataFrame(rows)
    suffix = f"instance{args.instance}" if args.instance else f"{len(indices)}instances"
    base = f"result/model2_pyomo_{args.case_name}_{suffix}"
    df.to_csv(base + ".csv", index=False)
    np.savez(base + ".npz",
             v_optimal=np.asarray(Vs), pg_optimal=np.asarray(PGs),
             qg_optimal=np.asarray(QGs),
             instance=np.asarray([i+1 for i in indices]),
             status=df["Status"].to_numpy(), success=df["Success"].to_numpy(),
             obj_val=df["Objective"].to_numpy(),
             solve_time=df["Solve_Time_s"].to_numpy(),
             max_eq=df["Max_Eq_Violation"].to_numpy(),
             max_ineq=df["Max_Ineq_Violation"].to_numpy())

    n_ok = int(df["Success"].sum())
    print("\n" + "="*68)
    print("MODEL 2 PYOMO/IPOPT SUMMARY")
    print("="*68)
    print(f"Successful solves: {n_ok}/{len(df)} ({100*n_ok/len(df):.2f}%)")
    if n_ok:
        good = df[df["Success"]]
        print(f"Mean solve time: {good['Solve_Time_s'].mean():.6f} s")
        print(f"Max Eq violation: {good['Max_Eq_Violation'].max():.3e}")
        print(f"Max Ineq violation: {good['Max_Ineq_Violation'].max():.3e}")
    print(f"Saved: {base}.csv")
    print(f"Saved: {base}.npz")


if __name__ == "__main__":
    main()
