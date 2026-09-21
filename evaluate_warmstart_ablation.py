#!/usr/bin/env python3
"""
Evaluate Ablated PINN dispatch predictions as primal initializations for the
ORIGINAL matrix-QCQP Pyomo/IPOPT ACOPF formulation.

Intended for the 14-, 57-, and 162-bus cases, for which the complete cold
IPOPT reference has already been saved as:

    result/warmstart_cold_case{bus_number}.csv

Important:
- Evaluates ONLY the Ablated PINN.
- Loads the CURRENT dataset and preserves its held-out test split.
- Reconstructs the matrix-QCQP representation in memory.
- Imports build_acopf_model() from pyomo_ipopt_qcqp.py unchanged.
- DOES NOT recompute cold IPOPT.
- Reuses the exact successful cold-reference instances, cold objective values,
  and cold runtimes from the existing warmstart_cold_case*.csv file.
- Uses Ablated PINN pg/qg only as the IPOPT primal initialization.
- Clips pg/qg to generator bounds, exactly as in the main warm-start study.
- Voltage remains at the same flat initialization as in the main study.
- Writes separate warmstart_ablation_* files and therefore does not overwrite
  the existing five-architecture warm-start results.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch

from pyomo_ipopt_qcqp import build_acopf_model
from ACOPF_ablation import AblatedQCQPMLP


# ============================================================
# Problem conversion / matrix reconstruction
# ============================================================

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
    Reconstruct the matrix representation required by build_acopf_model()
    from the graph/branch coefficients stored in the current dataset.
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

    M_pf = []
    M_qf = []
    M_pt = []
    M_qt = []

    # --------------------------------------------------------
    # Branch-flow matrices
    # --------------------------------------------------------

    for l in range(nbranch):
        i = int(fbus[l])
        j = int(tbus[l])

        iB = i + nbus
        jB = j + nbus

        # Active power at from end
        A = np.zeros((D, D))

        A[i, i] = g11[l]
        A[iB, iB] = g11[l]

        A[i, j] = -(g12[l] - b21[l])
        A[iB, jB] = -(g12[l] - b21[l])

        A[i, jB] = g21[l] + b12[l]
        A[iB, j] = -(g21[l] + b12[l])

        M_pf.append(0.5 * (A + A.T))

        # Reactive power at from end
        A = np.zeros((D, D))

        A[i, i] = -b11[l]
        A[iB, iB] = -b11[l]

        A[i, j] = b12[l] + g21[l]
        A[iB, jB] = b12[l] + g21[l]

        A[i, jB] = -(b21[l] - g12[l])
        A[iB, j] = b21[l] - g12[l]

        M_qf.append(0.5 * (A + A.T))

        # Active power at to end
        A = np.zeros((D, D))

        A[j, j] = g22[l]
        A[jB, jB] = g22[l]

        A[j, i] = -(g12[l] + b21[l])
        A[jB, iB] = -(g12[l] + b21[l])

        A[j, iB] = -(g21[l] - b12[l])
        A[jB, i] = g21[l] - b12[l]

        M_pt.append(0.5 * (A + A.T))

        # Reactive power at to end
        A = np.zeros((D, D))

        A[j, j] = -b22[l]
        A[jB, jB] = -b22[l]

        A[j, i] = b12[l] - g21[l]
        A[jB, iB] = b12[l] - g21[l]

        A[j, iB] = b21[l] + g12[l]
        A[jB, i] = -(b21[l] + g12[l])

        M_qt.append(0.5 * (A + A.T))

    # --------------------------------------------------------
    # Nodal active/reactive injection matrices
    # --------------------------------------------------------

    M_p = [
        np.zeros((D, D))
        for _ in range(nbus)
    ]

    M_q = [
        np.zeros((D, D))
        for _ in range(nbus)
    ]

    for i in range(nbus):
        M_p[i][i, i] = Gs[i]
        M_p[i][i + nbus, i + nbus] = Gs[i]

        M_q[i][i, i] = -Bs[i]
        M_q[i][i + nbus, i + nbus] = -Bs[i]

    for l in range(nbranch):
        i = int(fbus[l])
        j = int(tbus[l])

        M_p[i] += M_pf[l]
        M_q[i] += M_qf[l]

        M_p[j] += M_pt[l]
        M_q[j] += M_qt[l]

    # --------------------------------------------------------
    # Angle-difference matrices
    # --------------------------------------------------------

    M_c = []
    M_s = []

    for l in range(nbranch):
        i = int(fbus[l])
        j = int(tbus[l])

        iB = i + nbus
        jB = j + nbus

        A = np.zeros((D, D))

        A[i, j] = 1.0
        A[iB, jB] = 1.0

        M_c.append(
            0.5 * (A + A.T)
        )

        A = np.zeros((D, D))

        A[iB, j] = 1.0
        A[i, jB] = -1.0

        M_s.append(
            0.5 * (A + A.T)
        )

    # --------------------------------------------------------
    # Voltage-magnitude matrices
    # --------------------------------------------------------

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


# ============================================================
# Pyomo initialization utilities
# ============================================================

def set_loads(m, Pd, Qd):
    for b in m.BUS:
        m.Pd[b] = float(Pd[b])
        m.Qd[b] = float(Qd[b])


def set_nn_dispatch_start(
    m,
    problem,
    pg_pred,
    qg_pred,
    nbus,
):
    """
    Use the NN-predicted generator dispatch as the IPOPT initialization.

    As in the main warm-start experiment:
    - pg/qg are clipped to their physical generator bounds;
    - NN-predicted voltage is deliberately NOT used;
    - voltage starts from the standard flat start.
    """

    pmin = problem["pmin"]
    pmax = problem["pmax"]

    qmin = problem["qmin"]
    qmax = problem["qmax"]

    for g in m.GEN:
        m.pg[g].value = float(
            np.clip(
                pg_pred[g],
                pmin[g],
                pmax[g],
            )
        )

        m.qg[g].value = float(
            np.clip(
                qg_pred[g],
                qmin[g],
                qmax[g],
            )
        )

    for j in m.BUS2:
        m.v[j].value = (
            1.0
            if j < nbus
            else 0.0
        )


def success(results):
    return (
        results.solver.termination_condition
        in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.feasible,
        }
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case_name",
        required=True,
    )

    parser.add_argument(
        "--bus_number",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--total_samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--eval_limit",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--model_dir",
        default=None,
    )

    parser.add_argument(
        "--tee",
        action="store_true",
    )

    args = parser.parse_args()

    os.makedirs(
        "result",
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Load CURRENT dataset
    # ========================================================

    dataset_path = (
        f"./dataset/"
        f"{args.case_name}_"
        f"{args.total_samples}.pt"
    )

    print(
        f"Loading CURRENT dataset: "
        f"{dataset_path}"
    )

    problem_pt = torch.load(
        dataset_path,
        map_location="cpu",
    )

    problem_np = (
        reconstruct_qcqp_matrices(
            to_numpy_problem(
                problem_pt
            )
        )
    )

    print(
        "QCQP matrices reconstructed in memory."
    )

    nbus = int(
        problem_np["nbus"]
    )

    ngen = int(
        problem_np["ngen"]
    )

    slack_imag_idx = int(
        np.where(
            np.asarray(
                problem_np["a_ref"]
            )
            == 1
        )[0][0]
    )

    # ========================================================
    # Recover held-out test set
    # ========================================================

    n = int(
        problem_np[
            "Pd_all"
        ].shape[0]
    )

    train_size = int(
        0.8 * n
    )

    val_size = int(
        0.1 * n
    )

    test_start = (
        train_size
        +
        val_size
    )

    Pd_all = np.asarray(
        problem_np[
            "Pd_all"
        ][test_start:],
        dtype=float,
    )

    Qd_all = np.asarray(
        problem_np[
            "Qd_all"
        ][test_start:],
        dtype=float,
    )

    n_test = len(
        Pd_all
    )

    print(
        f"Held-out test instances: "
        f"{n_test}"
    )

    # ========================================================
    # Load EXISTING cold-IPOPT reference
    #
    # IMPORTANT:
    # NO cold IPOPT solve is performed in this script.
    # ========================================================

    cold_file = (
        f"result/"
        f"warmstart_cold_case"
        f"{args.bus_number}.csv"
    )

    if not os.path.exists(
        cold_file
    ):
        raise FileNotFoundError(
            "\nExisting cold-IPOPT reference "
            "not found:\n"
            f"    {cold_file}\n\n"
            "This evaluator is intended for "
            "the non-chunked 14/57/162-bus "
            "warm-start results."
        )

    print(
        "\n"
        + "=" * 68
    )

    print(
        "LOADING EXISTING COLD IPOPT BASELINE"
    )

    print(
        "=" * 68
    )

    print(
        f"Cold reference: "
        f"{cold_file}"
    )

    cold_df = pd.read_csv(
        cold_file
    )

    required_columns = {
        "Instance_ID",
        "Cold_IPOPT_Status",
        "Cold_IPOPT_Success",
        "Cold_IPOPT_Time_s",
        "Cold_IPOPT_Cost",
    }

    missing_columns = (
        required_columns
        -
        set(
            cold_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Cold reference CSV is "
            "missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Instance IDs
    # --------------------------------------------------------

    cold_df[
        "Instance_ID"
    ] = (
        cold_df[
            "Instance_ID"
        ].astype(int)
    )

    original_ids = (
        cold_df[
            "Instance_ID"
        ].to_numpy(
            dtype=int
        )
    )

    # Respect eval_limit if requested.
    if args.eval_limit is not None:

        keep = min(
            int(
                args.eval_limit
            ),
            len(
                original_ids
            ),
        )

        cold_df = (
            cold_df
            .iloc[:keep]
            .copy()
        )

        original_ids = (
            cold_df[
                "Instance_ID"
            ]
            .to_numpy(
                dtype=int
            )
        )

    # --------------------------------------------------------
    # Validate IDs against current held-out test set
    # --------------------------------------------------------

    if np.any(
        original_ids < 0
    ) or np.any(
        original_ids >= n_test
    ):
        raise ValueError(
            "Cold reference contains "
            "Instance_ID values outside "
            "the current held-out test set."
        )

    # The original Instance_ID is the index within the
    # 1,000-instance held-out test set.
    Pd = Pd_all[
        original_ids
    ]

    Qd = Qd_all[
        original_ids
    ]

    # --------------------------------------------------------
    # Robust cold-success conversion
    # --------------------------------------------------------

    success_col = (
        cold_df[
            "Cold_IPOPT_Success"
        ]
    )

    if success_col.dtype == bool:

        cold_success = (
            success_col
            .to_numpy()
        )

    else:

        cold_success = (
            success_col
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
            .to_numpy()
        )

    cold_times_all = (
        cold_df[
            "Cold_IPOPT_Time_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    cold_costs_all = (
        cold_df[
            "Cold_IPOPT_Cost"
        ]
        .to_numpy(
            dtype=float
        )
    )

    valid = (
        cold_success.copy()
    )

    n_requested = len(
        valid
    )

    n_valid = int(
        valid.sum()
    )

    n_failed = int(
        (~valid).sum()
    )

    print(
        f"Existing cold reference success: "
        f"{n_valid}/{n_requested} "
        f"("
        f"{100.0*n_valid/n_requested:.2f}"
        f"%)"
    )

    if n_failed > 0:

        failed_ids = (
            original_ids[
                ~valid
            ]
        )

        print(
            f"Cold IPOPT failed on "
            f"{n_failed} instance(s)."
        )

        print(
            "Failed Instance_ID(s): "
            f"{failed_ids.tolist()}"
        )

        print(
            "These instances are excluded "
            "from the paired Ablated-PINN "
            "restoration comparison."
        )

    if n_valid == 0:

        raise RuntimeError(
            "No successful cold-IPOPT "
            "reference instances are "
            "available."
        )

    # ========================================================
    # Apply EXACT SAME cold-feasible mask
    # ========================================================

    Pd = Pd[
        valid
    ]

    Qd = Qd[
        valid
    ]

    original_ids = (
        original_ids[
            valid
        ]
    )

    cold_times = (
        cold_times_all[
            valid
        ]
    )

    cold_costs = (
        cold_costs_all[
            valid
        ]
    )

    n_eval = (
        n_valid
    )

    print(
        f"Proceeding with "
        f"{n_eval} common "
        f"reference-feasible "
        f"instances."
    )

    # ========================================================
    # Build ORIGINAL matrix-QCQP model
    # ========================================================

    print(
        "\nBuilding ORIGINAL "
        "matrix-QCQP Pyomo model..."
    )

    m = build_acopf_model(
        problem_np,
        slack_imag_idx,
    )

    solver = (
        pyo.SolverFactory(
            "ipopt"
        )
    )

    solver.options[
        "tol"
    ] = 1e-6

    solver.options[
        "max_iter"
    ] = 3000

    solver.options[
        "max_cpu_time"
    ] = 30.0

    solver.options[
        "warm_start_init_point"
    ] = "no"

    print(
        "Model built successfully."
    )

    # ========================================================
    # Torch problem representation
    # ========================================================

    problem = {}

    for k, v in problem_pt.items():

        if isinstance(
            v,
            torch.Tensor,
        ):

            if (
                v.is_floating_point()
            ):

                problem[k] = (
                    v.to(
                        device=device,
                        dtype=torch.float32,
                    )
                )

            else:

                problem[k] = (
                    v.to(
                        device
                    )
                )

        else:

            problem[k] = v

    Pd_t = torch.tensor(
        Pd,
        dtype=torch.float32,
        device=device,
    )

    Qd_t = torch.tensor(
        Qd,
        dtype=torch.float32,
        device=device,
    )

    # ========================================================
    # Locate Ablated PINN checkpoints
    # ========================================================

    model_dir = (
        args.model_dir
        or
        os.path.join(
            "model",
            str(
                args.bus_number
            ),
        )
    )

    checkpoint_pattern = (
        os.path.join(
            model_dir,
            "*ablation_model*.pth",
        )
    )

    checkpoints = sorted(
        glob.glob(
            checkpoint_pattern
        )
    )

    print(
        "\n"
        + "=" * 68
    )

    print(
        "ABLATION CHECKPOINTS"
    )

    print(
        "=" * 68
    )

    print(
        f"Pattern: "
        f"{checkpoint_pattern}"
    )

    print(
        f"Found "
        f"{len(checkpoints)} "
        f"checkpoint(s)."
    )

    for ckpt in checkpoints:

        print(
            "  "
            + os.path.basename(
                ckpt
            )
        )

    if len(
        checkpoints
    ) == 0:

        raise FileNotFoundError(
            "No Ablated PINN "
            "checkpoints found using:\n"
            f"{checkpoint_pattern}"
        )

    if len(
        checkpoints
    ) != 5:

        print(
            "\nWARNING: Expected five "
            "independently trained "
            "Ablated PINN checkpoints, "
            f"but found "
            f"{len(checkpoints)}."
        )

    # ========================================================
    # Objective coefficients
    # ========================================================

    c2 = np.asarray(
        problem_np["c2"],
        dtype=float,
    )

    c1 = np.asarray(
        problem_np["c1"],
        dtype=float,
    )

    c0 = np.asarray(
        problem_np["c0"],
        dtype=float,
    )

    rows = []

    # ========================================================
    # Evaluate Ablated PINN
    # ========================================================

    print(
        "\n"
        + "=" * 68
    )

    print(
        "ABLATION PINN "
        "NN-INITIALIZED IPOPT"
    )

    print(
        "=" * 68
    )

    for run, ckpt in enumerate(
        checkpoints,
        1,
    ):

        print(
            f"\nRun {run}: "
            f"{os.path.basename(ckpt)}"
        )

        # ----------------------------------------------------
        # Instantiate model
        # ----------------------------------------------------

        net = AblatedQCQPMLP(
            nbus,
            ngen,
            slack_imag_idx,
        ).to(
            device
        )

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        sd = torch.load(
            ckpt,
            map_location=device,
            weights_only=True,
        )

        # Training used torch.compile(), which can add the
        # "_orig_mod." prefix to state-dict keys.
        sd = {
            k.replace(
                "_orig_mod.",
                "",
            ): v
            for k, v
            in sd.items()
        }

        net.load_state_dict(
            sd
        )

        net.float()
        net.eval()

        # ----------------------------------------------------
        # NN inference
        # ----------------------------------------------------

        with torch.no_grad():

            if (
                device.type
                == "cuda"
            ):
                torch.cuda.synchronize()

            t0 = (
                time.perf_counter()
            )

            out = net(
                Pd_t,
                Qd_t,
                problem,
            )

            if (
                device.type
                == "cuda"
            ):
                torch.cuda.synchronize()

            nn_total = (
                time.perf_counter()
                -
                t0
            )

        pg_all = (
            out[1]
            .detach()
            .cpu()
            .numpy()
        )

        qg_all = (
            out[2]
            .detach()
            .cpu()
            .numpy()
        )

        nn_per_instance = (
            nn_total
            /
            n_eval
        )

        print(
            f"NN inference: "
            f"{nn_per_instance:.9f} "
            f"s/instance"
        )

        # ----------------------------------------------------
        # NN-initialized IPOPT restoration
        # ----------------------------------------------------

        for i in range(
            n_eval
        ):

            set_loads(
                m,
                Pd[i],
                Qd[i],
            )

            set_nn_dispatch_start(
                m,
                problem_np,
                pg_all[i],
                qg_all[i],
                nbus,
            )

            t0 = (
                time.perf_counter()
            )

            try:

                res = solver.solve(
                    m,
                    tee=args.tee,
                )

                solve_time = (
                    time.perf_counter()
                    -
                    t0
                )

                ok = success(
                    res
                )

                term = str(
                    res.solver
                    .termination_condition
                )

            except Exception as exc:

                solve_time = (
                    time.perf_counter()
                    -
                    t0
                )

                ok = False

                term = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            # ------------------------------------------------
            # Feasible post-restoration cost
            # ------------------------------------------------

            if ok:

                pg_sol = np.array(
                    [
                        pyo.value(
                            m.pg[g]
                        )
                        for g
                        in m.GEN
                    ],
                    dtype=float,
                )

                cost = float(
                    np.sum(
                        c2
                        * pg_sol**2
                        +
                        c1
                        * pg_sol
                        +
                        c0
                    )
                )

                gap = (
                    100.0
                    *
                    (
                        cost
                        -
                        cold_costs[i]
                    )
                    /
                    cold_costs[i]
                )

            else:

                cost = np.nan
                gap = np.nan

            total = (
                nn_per_instance
                +
                solve_time
            )

            rows.append({
                "Architecture":
                    "Ablated PINN",

                "Run":
                    run,

                "Instance_ID":
                    int(
                        original_ids[i]
                    ),

                "Success":
                    ok,

                "Status":
                    term,

                "NN_Time_s":
                    nn_per_instance,

                "NN_Initialized_IPOPT_Time_s":
                    solve_time,

                "Total_Online_Time_s":
                    total,

                "Cold_IPOPT_Time_s":
                    cold_times[i],

                "Cold_IPOPT_Cost":
                    cold_costs[i],

                "Speedup_vs_Cold_IPOPT":
                    (
                        cold_times[i]
                        /
                        total
                        if total > 0
                        else np.nan
                    ),

                "Feasible_Cost":
                    cost,

                "Feasible_Obj_Gap_pct":
                    gap,
            })

            if (
                (not ok)
                or
                i == 0
                or
                (i + 1) % 25 == 0
                or
                (i + 1) == n_eval
            ):

                if ok:

                    print(
                        f"  "
                        f"[{i+1:4d}/{n_eval}] "
                        f"success=True "
                        f"status={term} "
                        f"time="
                        f"{solve_time:.6f}s "
                        f"gap="
                        f"{gap:+.4f}%"
                    )

                else:

                    print(
                        f"  "
                        f"[{i+1:4d}/{n_eval}] "
                        f"success=False "
                        f"status={term}"
                    )

        # ----------------------------------------------------
        # Incremental raw save after every completed run
        # ----------------------------------------------------

        raw_out = (
            f"result/"
            f"warmstart_ablation_raw_"
            f"case{args.bus_number}.csv"
        )

        pd.DataFrame(
            rows
        ).to_csv(
            raw_out,
            index=False,
        )

        print(
            f"Intermediate raw results "
            f"saved -> {raw_out}"
        )

    # ========================================================
    # Final aggregation
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    if len(df) == 0:

        raise RuntimeError(
            "No Ablated PINN evaluation "
            "rows were produced."
        )

    # --------------------------------------------------------
    # Run-level statistics
    # --------------------------------------------------------

    run_df = (
        df.groupby(
            [
                "Architecture",
                "Run",
            ],
            as_index=False,
        )
        .agg(
            N=(
                "Success",
                "size",
            ),

            Success_Rate=(
                "Success",
                "mean",
            ),

            NN_Time_s=(
                "NN_Time_s",
                "mean",
            ),

            IPOPT_Time_s=(
                "NN_Initialized_IPOPT_Time_s",
                "mean",
            ),

            Total_Online_Time_s=(
                "Total_Online_Time_s",
                "mean",
            ),

            Cold_IPOPT_Time_s=(
                "Cold_IPOPT_Time_s",
                "mean",
            ),

            Speedup=(
                "Speedup_vs_Cold_IPOPT",
                "mean",
            ),

            Feasible_Obj_Gap_pct=(
                "Feasible_Obj_Gap_pct",
                "mean",
            ),
        )
    )

    run_out = (
        f"result/"
        f"warmstart_ablation_runlevel_"
        f"case{args.bus_number}.csv"
    )

    run_df.to_csv(
        run_out,
        index=False,
    )

    # --------------------------------------------------------
    # Across-run summary
    # --------------------------------------------------------

    summary_df = (
        run_df
        .groupby(
            "Architecture"
        )
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    summary_out = (
        f"result/"
        f"warmstart_ablation_summary_"
        f"case{args.bus_number}.csv"
    )

    summary_df.to_csv(
        summary_out
    )

    # ========================================================
    # Console summary
    # ========================================================

    print(
        "\n"
        + "=" * 68
    )

    print(
        "ABLATION WARM-START "
        "RUN-LEVEL SUMMARY"
    )

    print(
        "=" * 68
    )

    print(
        run_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "  "
        f"result/"
        f"warmstart_ablation_raw_"
        f"case{args.bus_number}.csv"
    )

    print(
        "  "
        f"{run_out}"
    )

    print(
        "  "
        f"{summary_out}"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()