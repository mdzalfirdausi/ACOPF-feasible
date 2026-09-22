#!/usr/bin/env python3
"""
Post-restoration ACOPF evaluation for reviewer feasibility-optimality-runtime analysis.

For each trained neural model/run and each held-out test instance:
  1) obtain the raw NN prediction (v, pg, qg),
  2) solve a feasible ACOPF restoration problem with IPOPT that minimizes
     normalized squared distance to the NN prediction,
  3) report raw objective gap/violations, restored objective gap,
     restoration time, total online time, and restoration success.

This script does NOT retrain any model.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import time
import sys

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def generation_cost_np(pg, problem_np):
    pg = np.asarray(pg, dtype=float)
    c2 = np.asarray(problem_np["c2"], dtype=float)
    c1 = np.asarray(problem_np["c1"], dtype=float)
    c0 = np.asarray(problem_np["c0"], dtype=float)
    return float(np.sum(c2 * pg**2 + c1 * pg + c0))


def raw_metrics_torch(v, pg, qg, Pd, Qd, problem):
    """Return raw gap ingredients and constraint violations for a batch."""
    B = Pd.shape[0]
    nbus = int(problem["nbus"])

    smax = problem["smax"].unsqueeze(0)
    angmax = problem["angmax"].unsqueeze(0)
    angmin = problem["angmin"].unsqueeze(0)
    Vmin = problem["Vmin"].unsqueeze(0)
    Vmax = problem["Vmax"].unsqueeze(0)
    pmax = problem["pmax"].unsqueeze(0)
    pmin = problem["pmin"].unsqueeze(0)
    qmax = problem["qmax"].unsqueeze(0)
    qmin = problem["qmin"].unsqueeze(0)

    fbus = problem["fbus"]
    tbus = problem["tbus"]
    g11, g12 = problem["g11"].unsqueeze(0), problem["g12"].unsqueeze(0)
    g21, g22 = problem["g21"].unsqueeze(0), problem["g22"].unsqueeze(0)
    b11, b12 = problem["b11"].unsqueeze(0), problem["b12"].unsqueeze(0)
    b21, b22 = problem["b21"].unsqueeze(0), problem["b22"].unsqueeze(0)
    Gs, Bs = problem["Gs"].unsqueeze(0), problem["Bs"].unsqueeze(0)

    vr = v[:, :nbus]
    vi = v[:, nbus:]
    vv = vr**2 + vi**2

    vr_f, vi_f = vr[:, fbus], vi[:, fbus]
    vr_t, vi_t = vr[:, tbus], vi[:, tbus]
    vv_f = vr_f**2 + vi_f**2
    vv_t = vr_t**2 + vi_t**2
    v_rt_cross = vr_f * vr_t + vi_f * vi_t
    v_it_cross = vr_f * vi_t - vi_f * vr_t

    pf = g11 * vv_f - (g12 - b21) * v_rt_cross + (g21 + b12) * v_it_cross
    qf = -b11 * vv_f + (b12 + g21) * v_rt_cross + (b21 - g12) * v_it_cross
    pt = g22 * vv_t - (g12 + b21) * v_rt_cross + (g21 - b12) * v_it_cross
    qt = -b22 * vv_t + (b12 - g21) * v_rt_cross - (b21 + g12) * v_it_cross

    vp = Gs.expand(B, -1) * vv
    vq = -Bs.expand(B, -1) * vv
    fbus_exp = fbus.unsqueeze(0).expand(B, -1)
    tbus_exp = tbus.unsqueeze(0).expand(B, -1)
    vp.scatter_add_(1, fbus_exp, pf)
    vp.scatter_add_(1, tbus_exp, pt)
    vq.scatter_add_(1, fbus_exp, qf)
    vq.scatter_add_(1, tbus_exp, qt)

    h_p = (pg @ problem["C_g"].T) - Pd - vp
    h_q = (qg @ problem["C_g"].T) - Qd - vq
    eq = torch.cat([h_p.abs(), h_q.abs()], dim=1)

    g_sf = (pf**2 + qf**2) - smax.expand(B, -1)**2
    g_st = (pt**2 + qt**2) - smax.expand(B, -1)**2
    g_pg_max = pg - pmax.expand(B, -1)
    g_pg_min = pmin.expand(B, -1) - pg
    g_qg_max = qg - qmax.expand(B, -1)
    g_qg_min = qmin.expand(B, -1) - qg
    g_ang_min = torch.tan(angmin.expand(B, -1)) * v_rt_cross - v_it_cross
    g_ang_max = v_it_cross - torch.tan(angmax.expand(B, -1)) * v_rt_cross
    g_v_max = vv - Vmax.expand(B, -1)**2
    g_v_min = Vmin.expand(B, -1)**2 - vv

    ineq = torch.cat([
        F.relu(g_sf), F.relu(g_st),
        F.relu(g_pg_max), F.relu(g_pg_min),
        F.relu(g_qg_max), F.relu(g_qg_min),
        F.relu(g_ang_min), F.relu(g_ang_max),
        F.relu(g_v_max), F.relu(g_v_min),
    ], dim=1)

    return {
        "max_eq": eq.max(dim=1).values,
        "mean_eq": eq.mean(dim=1),
        "max_ineq": ineq.max(dim=1).values,
        "mean_ineq": ineq.mean(dim=1),
    }


def build_sparse_restoration_model(problem_np, slack_imag_idx):
    """Build the restoration NLP from the same sparse AC equations used by evaluate_new.py."""
    print("Building sparse Pyomo restoration model (this happens only once)...")

    nbus = int(problem_np["nbus"])
    ngen = int(problem_np["ngen"])
    nbranch = int(problem_np["nbranch"])

    fbus = np.asarray(problem_np["fbus"], dtype=int)
    tbus = np.asarray(problem_np["tbus"], dtype=int)
    C_g = np.asarray(problem_np["C_g"], dtype=float)

    Gs = np.asarray(problem_np["Gs"], dtype=float)
    Bs = np.asarray(problem_np["Bs"], dtype=float)
    g11 = np.asarray(problem_np["g11"], dtype=float)
    g12 = np.asarray(problem_np["g12"], dtype=float)
    g21 = np.asarray(problem_np["g21"], dtype=float)
    g22 = np.asarray(problem_np["g22"], dtype=float)
    b11 = np.asarray(problem_np["b11"], dtype=float)
    b12 = np.asarray(problem_np["b12"], dtype=float)
    b21 = np.asarray(problem_np["b21"], dtype=float)
    b22 = np.asarray(problem_np["b22"], dtype=float)

    pmin = np.asarray(problem_np["pmin"], dtype=float)
    pmax = np.asarray(problem_np["pmax"], dtype=float)
    qmin = np.asarray(problem_np["qmin"], dtype=float)
    qmax = np.asarray(problem_np["qmax"], dtype=float)
    Vmin = np.asarray(problem_np["Vmin"], dtype=float)
    Vmax = np.asarray(problem_np["Vmax"], dtype=float)
    smax = np.asarray(problem_np["smax"], dtype=float)
    angmin = np.asarray(problem_np["angmin"], dtype=float)
    angmax = np.asarray(problem_np["angmax"], dtype=float)

    m = pyo.ConcreteModel()
    m.BUS = pyo.RangeSet(0, nbus - 1)
    m.GEN = pyo.RangeSet(0, ngen - 1)
    m.BR = pyo.RangeSet(0, nbranch - 1)
    m.BUS2 = pyo.RangeSet(0, 2 * nbus - 1)

    # Hot-swappable loads and NN reference point.
    m.Pd = pyo.Param(m.BUS, mutable=True, initialize=0.0)
    m.Qd = pyo.Param(m.BUS, mutable=True, initialize=0.0)
    m.pg_nn = pyo.Param(m.GEN, mutable=True, initialize=0.0)
    m.qg_nn = pyo.Param(m.GEN, mutable=True, initialize=0.0)
    m.v_nn = pyo.Param(m.BUS2, mutable=True, initialize=0.0)

    m.pg = pyo.Var(
        m.GEN,
        bounds=lambda mm, g: (float(pmin[g]), float(pmax[g])),
        initialize=lambda mm, g: float((pmin[g] + pmax[g]) / 2.0),
    )
    m.qg = pyo.Var(
        m.GEN,
        bounds=lambda mm, g: (float(qmin[g]), float(qmax[g])),
        initialize=lambda mm, g: float((qmin[g] + qmax[g]) / 2.0),
    )

    vmax_rect = float(np.max(Vmax))
    m.v = pyo.Var(
        m.BUS2,
        bounds=(-vmax_rect, vmax_rect),
        initialize=lambda mm, j: 1.0 if j < nbus else 0.0,
    )
    m.v[int(slack_imag_idx)].fix(0.0)

    # Branch quantities: EXACT algebra used in evaluate_new.py.
    def vv(mm, i):
        return mm.v[i] ** 2 + mm.v[i + nbus] ** 2

    def cft(mm, l):
        i, j = int(fbus[l]), int(tbus[l])
        return mm.v[i] * mm.v[j] + mm.v[i + nbus] * mm.v[j + nbus]

    def sft(mm, l):
        i, j = int(fbus[l]), int(tbus[l])
        return mm.v[i] * mm.v[j + nbus] - mm.v[i + nbus] * mm.v[j]

    def pf_rule(mm, l):
        i = int(fbus[l])
        return (float(g11[l]) * vv(mm, i)
                - float(g12[l] - b21[l]) * cft(mm, l)
                + float(g21[l] + b12[l]) * sft(mm, l))

    def qf_rule(mm, l):
        i = int(fbus[l])
        return (-float(b11[l]) * vv(mm, i)
                + float(b12[l] + g21[l]) * cft(mm, l)
                + float(b21[l] - g12[l]) * sft(mm, l))

    def pt_rule(mm, l):
        j = int(tbus[l])
        return (float(g22[l]) * vv(mm, j)
                - float(g12[l] + b21[l]) * cft(mm, l)
                + float(g21[l] - b12[l]) * sft(mm, l))

    def qt_rule(mm, l):
        j = int(tbus[l])
        return (-float(b22[l]) * vv(mm, j)
                + float(b12[l] - g21[l]) * cft(mm, l)
                - float(b21[l] + g12[l]) * sft(mm, l))

    m.pf = pyo.Expression(m.BR, rule=pf_rule)
    m.qf = pyo.Expression(m.BR, rule=qf_rule)
    m.pt = pyo.Expression(m.BR, rule=pt_rule)
    m.qt = pyo.Expression(m.BR, rule=qt_rule)

    outgoing = {i: [] for i in range(nbus)}
    incoming = {i: [] for i in range(nbus)}
    gens_at_bus = {i: [] for i in range(nbus)}
    for l in range(nbranch):
        outgoing[int(fbus[l])].append(l)
        incoming[int(tbus[l])].append(l)
    for i in range(nbus):
        gens_at_bus[i] = [g for g in range(ngen) if C_g[i, g] != 0.0]

    m.cons = pyo.ConstraintList()

    # Nodal P/Q balance and voltage magnitude bounds.
    for i in range(nbus):
        gen_p = sum(float(C_g[i, g]) * m.pg[g] for g in gens_at_bus[i])
        gen_q = sum(float(C_g[i, g]) * m.qg[g] for g in gens_at_bus[i])

        # Same scatter-add convention as evaluate_new.py:
        # vp = Gs*|V|^2 + sum(from pf) + sum(to pt)
        # vq = -Bs*|V|^2 + sum(from qf) + sum(to qt)
        vp_i = (float(Gs[i]) * vv(m, i)
                + sum(m.pf[l] for l in outgoing[i])
                + sum(m.pt[l] for l in incoming[i]))
        vq_i = (-float(Bs[i]) * vv(m, i)
                + sum(m.qf[l] for l in outgoing[i])
                + sum(m.qt[l] for l in incoming[i]))

        m.cons.add(gen_p - m.Pd[i] == vp_i)
        m.cons.add(gen_q - m.Qd[i] == vq_i)
        m.cons.add(float(Vmin[i] ** 2) <= vv(m, i))
        m.cons.add(vv(m, i) <= float(Vmax[i] ** 2))

    # Same thermal and angle inequalities used by evaluate_new.py.
    for l in range(nbranch):
        m.cons.add(m.pf[l] ** 2 + m.qf[l] ** 2 <= float(smax[l] ** 2))
        m.cons.add(m.pt[l] ** 2 + m.qt[l] ** 2 <= float(smax[l] ** 2))
        m.cons.add(float(np.tan(angmin[l])) * cft(m, l) <= sft(m, l))
        m.cons.add(sft(m, l) <= float(np.tan(angmax[l])) * cft(m, l))

    # Normalized distance to the raw neural prediction.
    p_scale = np.maximum(pmax - pmin, 1e-6)
    q_scale = np.maximum(qmax - qmin, 1e-6)
    v_scale = max(vmax_rect, 1e-6)

    def restoration_rule(mm):
        pg_term = sum(((mm.pg[g] - mm.pg_nn[g]) / float(p_scale[g])) ** 2 for g in range(ngen))
        qg_term = sum(((mm.qg[g] - mm.qg_nn[g]) / float(q_scale[g])) ** 2 for g in range(ngen))
        v_term = sum(((mm.v[j] - mm.v_nn[j]) / v_scale) ** 2 for j in range(2 * nbus))
        return pg_term + qg_term + v_term

    m.restoration_obj = pyo.Objective(rule=restoration_rule, sense=pyo.minimize)
    print("Sparse restoration model built successfully.")
    return m

def is_success(results):
    tc = results.solver.termination_condition
    return tc in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    }


def configure_instance(m, Pd, Qd, v_pred, pg_pred, qg_pred, problem_np, slack_imag_idx):
    nbus = int(problem_np["nbus"])
    ngen = int(problem_np["ngen"])
    pmin = np.asarray(problem_np["pmin"], dtype=float)
    pmax = np.asarray(problem_np["pmax"], dtype=float)
    qmin = np.asarray(problem_np["qmin"], dtype=float)
    qmax = np.asarray(problem_np["qmax"], dtype=float)
    vmax_rect = float(np.max(problem_np["Vmax"]))

    for b in range(nbus):
        m.Pd[b] = float(Pd[b])
        m.Qd[b] = float(Qd[b])

    # Reference point = raw NN prediction. Initial point = clipped prediction.
    for g in range(ngen):
        m.pg_nn[g] = float(pg_pred[g])
        m.qg_nn[g] = float(qg_pred[g])
        m.pg[g].value = float(np.clip(pg_pred[g], pmin[g], pmax[g]))
        m.qg[g].value = float(np.clip(qg_pred[g], qmin[g], qmax[g]))

    for j in range(2 * nbus):
        ref = 0.0 if j == slack_imag_idx else float(v_pred[j])
        m.v_nn[j] = ref
        if j != slack_imag_idx:
            m.v[j].value = float(np.clip(v_pred[j], -vmax_rect, vmax_rect))


def main():
    parser = argparse.ArgumentParser(description="Post-restoration ACOPF evaluation.")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--bus_number", type=int, required=True)
    parser.add_argument("--total_samples", type=int, default=10000)
    parser.add_argument("--eval_limit", type=int, default=1000)
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Checkpoint directory. Default: ./model/<bus_number>")
    parser.add_argument("--ipopt_tol", type=float, default=1e-6)
    parser.add_argument("--ipopt_max_iter", type=int, default=3000)
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()

    os.makedirs("result", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Prediction device: {device}")

    case_name = args.case_name
    bus_number = args.bus_number
    model_dir = args.model_dir or os.path.join("model", str(bus_number))
    print(f"Model directory: {model_dir}")

    dataset_path = f"./dataset/{case_name}_{args.total_samples}.pt"
    print(f"Loading dataset: {dataset_path}")
    problem_pt_cpu = torch.load(dataset_path, map_location="cpu")
    problem_np = to_numpy_problem(problem_pt_cpu)

    actual_total_samples = int(problem_np["Pd_all"].shape[0])
    train_size = int(0.8 * actual_total_samples)
    val_size = int(0.1 * actual_total_samples)
    test_start = train_size + val_size

    test_Pd_all = np.asarray(problem_np["Pd_all"][test_start:], dtype=np.float64)
    test_Qd_all = np.asarray(problem_np["Qd_all"][test_start:], dtype=np.float64)

    gt_path = f"./result/ipopt_baseline_{case_name}_{actual_total_samples - test_start}_instances.npz"
    print(f"Loading IPOPT ground truth: {gt_path}")
    gt_data = np.load(gt_path)
    status = gt_data["status"]
    mask = np.array([("ok" in str(s).lower()) or ("optimal" in str(s).lower()) for s in status])
    valid_original_ids = np.where(mask)[0]

    test_Pd = test_Pd_all[mask]
    test_Qd = test_Qd_all[mask]
    test_v_gt = np.asarray(gt_data["v_optimal"][mask], dtype=np.float64)
    test_pg_gt = np.asarray(gt_data["pg_optimal"][mask], dtype=np.float64)
    test_qg_gt = np.asarray(gt_data["qg_optimal"][mask], dtype=np.float64)

    n_eval = min(args.eval_limit, len(test_Pd))
    test_Pd = test_Pd[:n_eval]
    test_Qd = test_Qd[:n_eval]
    test_v_gt = test_v_gt[:n_eval]
    test_pg_gt = test_pg_gt[:n_eval]
    test_qg_gt = test_qg_gt[:n_eval]
    valid_original_ids = valid_original_ids[:n_eval]

    nbus = int(problem_np["nbus"])
    ngen = int(problem_np["ngen"])
    nbranch = int(problem_np["nbranch"])
    slack_imag_idx = int(np.where(np.asarray(problem_np["a_ref"]) == 1)[0][0])

    # Torch copy for NN prediction and raw-violation evaluation.
    problem = {}
    for key, value in problem_pt_cpu.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                problem[key] = value.to(device=device, dtype=torch.float32)
            else:
                problem[key] = value.to(device=device)
        else:
            problem[key] = value

    Pd_t = torch.tensor(test_Pd, dtype=torch.float32, device=device)
    Qd_t = torch.tensor(test_Qd, dtype=torch.float32, device=device)
    pg_gt_t = torch.tensor(test_pg_gt, dtype=torch.float32, device=device)

    c2 = problem["c2"].unsqueeze(0)
    c1 = problem["c1"].unsqueeze(0)
    c0 = problem["c0"].unsqueeze(0)
    ipopt_cost_t = (c2 * pg_gt_t**2 + c1 * pg_gt_t + c0).sum(dim=1)
    ipopt_cost = ipopt_cost_t.detach().cpu().numpy()

    def get_model_paths(keyword):
        paths = sorted(glob.glob(os.path.join(model_dir, f"*{keyword}*.pth")))
        if not paths:
            print(f"WARNING: no checkpoints matching '{keyword}' in {model_dir}")
        return paths

    architectures = {
        "PINN Baseline": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": get_model_paths("pinn_model"),
        },
        "DC3": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": get_model_paths("dc3_model"),
        },
        "FSNet": {
            "class": lambda: baselineQCQPMLP(nbus, ngen, slack_imag_idx).to(device),
            "paths": get_model_paths("fsnet_model"),
        },
        "KKT": {
            "class": lambda: HardKKT_QCQPMLP(nbus, ngen, nbranch, slack_imag_idx).to(device),
            "paths": get_model_paths("hardkkt"),
        },
        "Rahul's Model": {
            "class": lambda: RahulSinglePINN_Smax(nbus, ngen, nbranch).to(device),
            "paths": get_model_paths("rahul_model"),
        },
    }

    # Build the restoration NLP once and hot-swap loads/reference predictions.
    restoration_model = build_sparse_restoration_model(problem_np, slack_imag_idx)
    solver = pyo.SolverFactory("ipopt")
    if not solver.available(False):
        raise RuntimeError("IPOPT is not available. Ensure `ipopt` is installed and on PATH.")
    solver.options["tol"] = args.ipopt_tol
    solver.options["max_iter"] = args.ipopt_max_iter
    solver.options["warm_start_init_point"] = "yes"

    rows = []

    for arch_name, cfg in architectures.items():
        if not cfg["paths"]:
            continue

        print(f"\n================ {arch_name} ================")
        for run_idx, ckpt in enumerate(cfg["paths"], start=1):
            print(f"\nRun {run_idx}: {os.path.basename(ckpt)}")
            model = cfg["class"]()
            state_dict = torch.load(ckpt, map_location=device, weights_only=True)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model = model.to(device).float().eval()

            # Predict all selected test instances in one batch, matching evaluate_new.py.
            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                outputs = model(Pd_t, Qd_t, problem)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_total = time.perf_counter() - t0

                v_pred_t, pg_pred_t, qg_pred_t = outputs[0], outputs[1], outputs[2]
                raw_viol = raw_metrics_torch(v_pred_t, pg_pred_t, qg_pred_t, Pd_t, Qd_t, problem)
                raw_cost_t = (c2 * pg_pred_t**2 + c1 * pg_pred_t + c0).sum(dim=1)
                raw_gap_t = 100.0 * (raw_cost_t - ipopt_cost_t) / ipopt_cost_t

            inference_per_instance = inference_total / n_eval
            v_pred_all = v_pred_t.detach().cpu().numpy().astype(np.float64)
            pg_pred_all = pg_pred_t.detach().cpu().numpy().astype(np.float64)
            qg_pred_all = qg_pred_t.detach().cpu().numpy().astype(np.float64)
            raw_gap = raw_gap_t.detach().cpu().numpy()
            raw_max_eq = raw_viol["max_eq"].detach().cpu().numpy()
            raw_mean_eq = raw_viol["mean_eq"].detach().cpu().numpy()
            raw_max_ineq = raw_viol["max_ineq"].detach().cpu().numpy()
            raw_mean_ineq = raw_viol["mean_ineq"].detach().cpu().numpy()

            for i in range(n_eval):
                configure_instance(
                    restoration_model,
                    test_Pd[i], test_Qd[i],
                    v_pred_all[i], pg_pred_all[i], qg_pred_all[i],
                    problem_np, slack_imag_idx,
                )

                start = time.perf_counter()
                try:
                    results = solver.solve(restoration_model, tee=args.tee)
                    restoration_time = time.perf_counter() - start
                    success = is_success(results)
                    term = str(results.solver.termination_condition)
                except Exception as exc:
                    restoration_time = time.perf_counter() - start
                    success = False
                    term = f"exception: {type(exc).__name__}: {exc}"

                if success:
                    v_rest = np.array([pyo.value(restoration_model.v[j]) for j in restoration_model.BUS2], dtype=float)
                    pg_rest = np.array([pyo.value(restoration_model.pg[g]) for g in restoration_model.GEN], dtype=float)
                    qg_rest = np.array([pyo.value(restoration_model.qg[g]) for g in restoration_model.GEN], dtype=float)
                    restored_cost = generation_cost_np(pg_rest, problem_np)
                    restored_gap = 100.0 * (restored_cost - ipopt_cost[i]) / ipopt_cost[i]
                    restoration_distance = float(pyo.value(restoration_model.restoration_obj))

                    # Check restored residuals with exactly the same metric code.
                    with torch.no_grad():
                        v_r_t = torch.tensor(v_rest[None, :], dtype=torch.float32, device=device)
                        pg_r_t = torch.tensor(pg_rest[None, :], dtype=torch.float32, device=device)
                        qg_r_t = torch.tensor(qg_rest[None, :], dtype=torch.float32, device=device)
                        pd_i_t = Pd_t[i:i+1]
                        qd_i_t = Qd_t[i:i+1]
                        rv = raw_metrics_torch(v_r_t, pg_r_t, qg_r_t, pd_i_t, qd_i_t, problem)
                        restored_max_eq = float(rv["max_eq"].item())
                        restored_mean_eq = float(rv["mean_eq"].item())
                        restored_max_ineq = float(rv["max_ineq"].item())
                        restored_mean_ineq = float(rv["mean_ineq"].item())
                else:
                    restored_cost = np.nan
                    restored_gap = np.nan
                    restoration_distance = np.nan
                    restored_max_eq = np.nan
                    restored_mean_eq = np.nan
                    restored_max_ineq = np.nan
                    restored_mean_ineq = np.nan

                rows.append({
                    "Architecture": arch_name,
                    "Run": run_idx,
                    "Instance_ID": int(valid_original_ids[i]),
                    "Raw_Obj_Gap_pct": float(raw_gap[i]),
                    "Raw_Max_Eq": float(raw_max_eq[i]),
                    "Raw_Mean_Eq": float(raw_mean_eq[i]),
                    "Raw_Max_Ineq": float(raw_max_ineq[i]),
                    "Raw_Mean_Ineq": float(raw_mean_ineq[i]),
                    "NN_Time_s": float(inference_per_instance),
                    "Restoration_Success": bool(success),
                    "Restoration_Status": term,
                    "Restoration_Time_s": float(restoration_time),
                    "Total_Online_Time_s": float(inference_per_instance + restoration_time),
                    "Restoration_Distance": restoration_distance,
                    "Restored_Cost": restored_cost,
                    "IPOPT_Cost": float(ipopt_cost[i]),
                    "Restored_Obj_Gap_pct": restored_gap,
                    "Restored_Max_Eq": restored_max_eq,
                    "Restored_Mean_Eq": restored_mean_eq,
                    "Restored_Max_Ineq": restored_max_ineq,
                    "Restored_Mean_Ineq": restored_mean_ineq,
                })

                if (i + 1) % 25 == 0 or i == 0 or (i + 1) == n_eval:
                    print(
                        f"  [{i+1:4d}/{n_eval}] success={success} "
                        f"raw_gap={raw_gap[i]:+.2f}% "
                        f"rest_gap={restored_gap:+.2f}% " if success else
                        f"  [{i+1:4d}/{n_eval}] success=False status={term}"
                    )

            # Save incrementally after every run so long experiments are recoverable.
            df_partial = pd.DataFrame(rows)
            raw_out = f"result/restoration_raw_case{bus_number}.csv"
            df_partial.to_csv(raw_out, index=False)
            print(f"Saved progress -> {raw_out}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        print("No model checkpoints were evaluated.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    raw_out = f"result/restoration_raw_case{bus_number}.csv"
    df.to_csv(raw_out, index=False)

    # Per-run summaries preserve the independent training-run structure.
    def summarize_run(g):
        ok = g["Restoration_Success"].astype(bool)
        valid = g.loc[ok]
        return pd.Series({
            "N": len(g),
            "Restoration_Success_Rate": ok.mean(),
            "Raw_Obj_Gap_Mean_pct": g["Raw_Obj_Gap_pct"].mean(),
            "Raw_Max_Eq_Mean": g["Raw_Max_Eq"].mean(),
            "Raw_Max_Ineq_Mean": g["Raw_Max_Ineq"].mean(),
            "Restored_Obj_Gap_Mean_pct": valid["Restored_Obj_Gap_pct"].mean(),
            "Restored_Abs_Obj_Gap_Mean_pct": valid["Restored_Obj_Gap_pct"].abs().mean(),
            "Restored_Max_Eq_Mean": valid["Restored_Max_Eq"].mean(),
            "Restored_Max_Ineq_Mean": valid["Restored_Max_Ineq"].mean(),
            "NN_Time_Mean_s": g["NN_Time_s"].mean(),
            "Restoration_Time_Mean_s": g["Restoration_Time_s"].mean(),
            "Total_Online_Time_Mean_s": g["Total_Online_Time_s"].mean(),
        })

    run_summary = (
        df.groupby(["Architecture", "Run"], sort=False)
          .apply(summarize_run, include_groups=False)
          .reset_index()
    )
    run_out = f"result/restoration_runlevel_case{bus_number}.csv"
    run_summary.to_csv(run_out, index=False)

    # Final architecture summary = mean +/- sample SD across independent runs.
    metrics = [
        "Restoration_Success_Rate",
        "Raw_Obj_Gap_Mean_pct",
        "Raw_Max_Eq_Mean",
        "Raw_Max_Ineq_Mean",
        "Restored_Obj_Gap_Mean_pct",
        "Restored_Abs_Obj_Gap_Mean_pct",
        "Restored_Max_Eq_Mean",
        "Restored_Max_Ineq_Mean",
        "NN_Time_Mean_s",
        "Restoration_Time_Mean_s",
        "Total_Online_Time_Mean_s",
    ]

    summary_rows = []
    for arch, g in run_summary.groupby("Architecture", sort=False):
        row = {"Architecture": arch, "Runs": len(g)}
        for col in metrics:
            row[f"{col}_Mean"] = g[col].mean()
            row[f"{col}_SD"] = g[col].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_out = f"result/restoration_summary_case{bus_number}.csv"
    summary.to_csv(summary_out, index=False)

    print("\n============================================================")
    print("POST-RESTORATION EVALUATION COMPLETE")
    print("============================================================")
    print(summary.to_string(index=False))
    print(f"\nRaw instance results : {raw_out}")
    print(f"Run-level results    : {run_out}")
    print(f"Architecture summary : {summary_out}")


if __name__ == "__main__":
    main()
