#!/usr/bin/env python3

import argparse
import time

import numpy as np
import pyomo.environ as pyo
import torch

from pyomo_ipopt_qcqp import build_acopf_model
from evaluate_warmstart_qcqp_currentbaseline import (
    reconstruct_qcqp_matrices,
    to_numpy_problem,
    set_loads,
    set_cold_start,
)


def quad_form(m, matrix):
    """
    Construct v^T M v using the same rectangular voltage vector
    used by pyomo_ipopt_qcqp.py.
    """
    rows, cols = np.nonzero(matrix)

    expr = 0.0

    for r, c in zip(rows, cols):
        value = float(matrix[r, c])

        if value != 0.0:
            expr += value * m.v[int(r)] * m.v[int(c)]

    return expr


def main():

    parser = argparse.ArgumentParser(
        description="Diagnose feasibility of one ACOPF test instance."
    )

    parser.add_argument(
        "--case_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--instance",
        type=int,
        required=True,
        help="Displayed test instance number, starting from 1.",
    )

    parser.add_argument(
        "--total_samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--tee",
        action="store_true",
    )

    args = parser.parse_args()

    # ============================================================
    # 1. LOAD CURRENT DATASET
    # ============================================================

    dataset_path = (
        f"./dataset/{args.case_name}_{args.total_samples}.pt"
    )

    print(f"Loading dataset: {dataset_path}")

    problem_pt = torch.load(
        dataset_path,
        map_location="cpu"
    )

    problem_np = to_numpy_problem(problem_pt)

    # Reconstruct the QCQP matrices from the CURRENT dataset
    problem_np = reconstruct_qcqp_matrices(problem_np)

    print("QCQP matrices reconstructed.")

    nbus = int(problem_np["nbus"])
    ngen = int(problem_np["ngen"])

    # ============================================================
    # 2. GET TEST SET
    # ============================================================

    n_total = int(problem_np["Pd_all"].shape[0])

    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)

    test_start = train_size + val_size

    test_Pd = np.asarray(
        problem_np["Pd_all"][test_start:],
        dtype=float
    )

    test_Qd = np.asarray(
        problem_np["Qd_all"][test_start:],
        dtype=float
    )

    # User uses displayed indexing: instance 46 -> Python index 45
    idx = args.instance - 1

    if idx < 0 or idx >= len(test_Pd):
        raise ValueError(
            f"Instance must be between 1 and {len(test_Pd)}."
        )

    Pd = test_Pd[idx]
    Qd = test_Qd[idx]

    print()
    print("=" * 70)
    print(f"DIAGNOSING TEST INSTANCE {args.instance}")
    print("=" * 70)

    print(f"Total Pd = {Pd.sum():.8f}")
    print(f"Total Qd = {Qd.sum():.8f}")

    print(
        f"Total P limits = "
        f"[{np.sum(problem_np['pmin']):.8f}, "
        f"{np.sum(problem_np['pmax']):.8f}]"
    )

    print(
        f"Total Q limits = "
        f"[{np.sum(problem_np['qmin']):.8f}, "
        f"{np.sum(problem_np['qmax']):.8f}]"
    )

    # ============================================================
    # 3. BUILD ORIGINAL MODEL
    # ============================================================

    slack_imag_idx = int(
        np.where(
            np.asarray(problem_np["a_ref"]) == 1
        )[0][0]
    )

    print("\nBuilding original QCQP ACOPF model...")

    m = build_acopf_model(
        problem_np,
        slack_imag_idx
    )

    set_loads(
        m,
        Pd,
        Qd
    )

    set_cold_start(
        m,
        problem_np,
        nbus
    )

    # ============================================================
    # 4. DEACTIVATE ORIGINAL P/Q BALANCE EQUALITIES
    # ============================================================

    equality_count = 0

    for con in m.component_data_objects(
        pyo.Constraint,
        active=True
    ):
        if con.equality:
            con.deactivate()
            equality_count += 1

    print(
        f"Deactivated {equality_count} original equality constraints."
    )

    expected_equalities = 2 * nbus

    if equality_count != expected_equalities:

        print(
            "\nWARNING:"
            f" expected {expected_equalities} P/Q equality constraints,"
            f" but found {equality_count}."
        )

        print(
            "Check build_acopf_model() before interpreting the result."
        )

    # ============================================================
    # 5. DEACTIVATE ECONOMIC OBJECTIVE
    # ============================================================

    m.cost.deactivate()

    # ============================================================
    # 6. CREATE POWER-BALANCE SLACK VARIABLES
    # ============================================================

    m.sp_pos = pyo.Var(
        m.BUS,
        domain=pyo.NonNegativeReals,
        initialize=0.0
    )

    m.sp_neg = pyo.Var(
        m.BUS,
        domain=pyo.NonNegativeReals,
        initialize=0.0
    )

    m.sq_pos = pyo.Var(
        m.BUS,
        domain=pyo.NonNegativeReals,
        initialize=0.0
    )

    m.sq_neg = pyo.Var(
        m.BUS,
        domain=pyo.NonNegativeReals,
        initialize=0.0
    )

    # ============================================================
    # 7. REBUILD RELAXED P/Q BALANCE EQUATIONS
    # ============================================================

    C_g = np.asarray(
        problem_np["C_g"],
        dtype=float
    )

    M_p = problem_np["M_p"]
    M_q = problem_np["M_q"]

    m.RelaxedPowerBalance = pyo.ConstraintList()

    for b in range(nbus):

        # --------------------------------------------------------
        # Active generation at bus
        # --------------------------------------------------------

        gen_p = sum(
            float(C_g[b, g]) * m.pg[g]
            for g in range(ngen)
            if C_g[b, g] != 0
        )

        vp = quad_form(
            m,
            M_p[b]
        )

        # Original:
        #
        # Cg Pg - Pd = v^T Mp v
        #
        # Relaxed:
        #
        # Cg Pg - Pd - v^T Mp v
        #     = sp_pos - sp_neg
        #

        m.RelaxedPowerBalance.add(
            gen_p
            - m.Pd[b]
            - vp
            ==
            m.sp_pos[b]
            - m.sp_neg[b]
        )

        # --------------------------------------------------------
        # Reactive generation at bus
        # --------------------------------------------------------

        gen_q = sum(
            float(C_g[b, g]) * m.qg[g]
            for g in range(ngen)
            if C_g[b, g] != 0
        )

        vq = quad_form(
            m,
            M_q[b]
        )

        m.RelaxedPowerBalance.add(
            gen_q
            - m.Qd[b]
            - vq
            ==
            m.sq_pos[b]
            - m.sq_neg[b]
        )

    # ============================================================
    # 8. FEASIBILITY OBJECTIVE
    # ============================================================

    m.feasibility_objective = pyo.Objective(

        expr=sum(
            m.sp_pos[b]
            + m.sp_neg[b]
            + m.sq_pos[b]
            + m.sq_neg[b]

            for b in m.BUS
        ),

        sense=pyo.minimize
    )

    # ============================================================
    # 9. SOLVE FEASIBILITY PROBLEM
    # ============================================================

    solver = pyo.SolverFactory("ipopt")

    solver.options["tol"] = 1e-8
    solver.options["acceptable_tol"] = 1e-7
    solver.options["max_iter"] = 5000
    solver.options["warm_start_init_point"] = "no"

    print()
    print("=" * 70)
    print("SOLVING MINIMUM POWER-BALANCE VIOLATION PROBLEM")
    print("=" * 70)

    start = time.perf_counter()

    results = solver.solve(
        m,
        tee=args.tee
    )

    elapsed = time.perf_counter() - start

    termination = results.solver.termination_condition

    # ============================================================
    # 10. EXTRACT SLACKS
    # ============================================================

    sp = np.array([
        pyo.value(m.sp_pos[b])
        + pyo.value(m.sp_neg[b])
        for b in m.BUS
    ])

    sq = np.array([
        pyo.value(m.sq_pos[b])
        + pyo.value(m.sq_neg[b])
        for b in m.BUS
    ])

    total_p_slack = float(sp.sum())
    total_q_slack = float(sq.sum())

    max_p_slack = float(sp.max())
    max_q_slack = float(sq.max())

    objective = pyo.value(
        m.feasibility_objective
    )

    # ============================================================
    # 11. REPORT
    # ============================================================

    print()
    print("=" * 70)
    print("FEASIBILITY DIAGNOSTIC RESULT")
    print("=" * 70)

    print(
        f"Instance                  : {args.instance}"
    )

    print(
        f"Termination               : {termination}"
    )

    print(
        f"Solve time                : {elapsed:.6f} s"
    )

    print(
        f"Feasibility objective     : {objective:.12e}"
    )

    print(
        f"Total active-power slack  : {total_p_slack:.12e}"
    )

    print(
        f"Total reactive-power slack: {total_q_slack:.12e}"
    )

    print(
        f"Max active-power slack    : {max_p_slack:.12e}"
    )

    print(
        f"Max reactive-power slack  : {max_q_slack:.12e}"
    )

    # Find worst buses

    worst_p_bus = int(np.argmax(sp))
    worst_q_bus = int(np.argmax(sq))

    print(
        f"Worst active bus index    : {worst_p_bus}"
    )

    print(
        f"Worst reactive bus index  : {worst_q_bus}"
    )

    print()
    print("-" * 70)

    # Conservative interpretation thresholds
    if objective <= 1e-6:

        print(
            "RESULT: A numerically feasible ACOPF point was found."
        )

        print(
            "The ordinary IPOPT failure is therefore most likely "
            "a nonlinear convergence/initialization issue."
        )

    elif objective <= 1e-4:

        print(
            "RESULT: The minimum violation is very small but not "
            "strictly zero."
        )

        print(
            "Treat this as near-feasible and inspect solver tolerances "
            "and the largest residuals."
        )

    else:

        print(
            "RESULT: A non-negligible power-balance relaxation remains."
        )

        print(
            "This is evidence that instance "
            f"{args.instance} may be infeasible under the current "
            "ACOPF constraints, or that IPOPT is still trapped at a "
            "local minimum of the feasibility-relaxation problem."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()