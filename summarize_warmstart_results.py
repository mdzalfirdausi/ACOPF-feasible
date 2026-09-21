#!/usr/bin/env python3
"""
summarize_warmstart_results.py

Post-process feasibility-restoration / warm-start ACOPF experiments.

Expected setup
--------------
14 and 57 bus:
    Non-chunked main warm-start files, e.g.
        result/warmstart_raw_case14.csv
        result/warmstart_raw_case57.csv

    Non-chunked ablation files, e.g.
        result/warmstart_ablation_raw_case14.csv
        result/warmstart_ablation_raw_case57.csv

162 and 300 bus:
    Chunked main warm-start files, e.g.
        result/warmstart_raw_case162_chunk000.csv
        ...
        result/warmstart_raw_case162_chunk099.csv

    Chunked ablation files, e.g.
        result/warmstart_ablation_raw_case162_chunk000.csv
        ...
        result/warmstart_ablation_raw_case162_chunk099.csv

The script:
    1. Finds and merges raw warm-start CSV files.
    2. Merges Ablated PINN results.
    3. Checks duplicate (Architecture, Run, Instance_ID) observations.
    4. Calculates run-level statistics.
    5. Calculates mean +/- SD across the five independent training runs.
    6. Computes 95% t confidence intervals across the five run-level values.
    7. Writes combined/raw, run-level, and final summary CSV files.
    8. Prints a paper-ready summary.

No solver or neural-network evaluation is performed.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import t


# ============================================================
# Configuration
# ============================================================

BUSES = [14, 57, 162, 300]

EXPECTED_RUNS = 5
EXPECTED_TEST_INSTANCES = 1000

RESULT_DIR = "result"


# ============================================================
# Helpers
# ============================================================

def read_csvs(files):
    """Read and concatenate a list of CSV files."""

    if not files:
        return pd.DataFrame()

    frames = []

    for filename in files:
        df = pd.read_csv(filename)

        # Keep source information for debugging only.
        df["_source_file"] = os.path.basename(filename)

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def find_main_raw_files(bus):
    """
    Find MAIN architecture raw warm-start files.

    Layout:
      14/57:
        result/warmstart_raw_caseXX.csv

      162:
        normally result/warmstart_raw_case162.csv
        or chunk files if available

      300:
        result/warmstart_ablation_case300_chunk/
            warmstart_raw_case300_chunk000.csv
            ...
    """

    # ========================================================
    # 1. Known chunk directories
    # ========================================================

    special_dirs = {
        162: os.path.join(
            RESULT_DIR,
            "warmstart_ablation_case162_chunk",
        ),
        300: os.path.join(
            RESULT_DIR,
            "warmstart_ablation_case300_chunk",
        ),
    }

    if bus in special_dirs:

        pattern = os.path.join(
            special_dirs[bus],
            f"warmstart_raw_case{bus}_chunk*.csv",
        )

        files = sorted(glob.glob(pattern))

        if files:
            print(
                f"Main chunk directory: {special_dirs[bus]}"
            )
            return files, True

    # ========================================================
    # 2. Chunk files directly inside result/
    # ========================================================

    pattern = os.path.join(
        RESULT_DIR,
        f"warmstart_raw_case{bus}_chunk*.csv",
    )

    files = sorted(glob.glob(pattern))

    if files:
        return files, True

    # ========================================================
    # 3. Non-chunked file
    # ========================================================

    full_file = os.path.join(
        RESULT_DIR,
        f"warmstart_raw_case{bus}.csv",
    )

    if os.path.exists(full_file):
        return [full_file], False

    return [], False


def find_ablation_raw_files(bus):
    """
    Find ABLATED PINN raw warm-start files.

    Layout:
      14/57:
        result/warmstart_ablation_raw_caseXX.csv

      162:
        result/warmstart_ablation_case162_chunk/

      300:
        result/warmstart_ablation_case300_chunk/
    """

    # ========================================================
    # 1. Known chunk directories
    # ========================================================

    special_dirs = {
        162: os.path.join(
            RESULT_DIR,
            "warmstart_ablation_case162_chunk",
        ),
        300: os.path.join(
            RESULT_DIR,
            "warmstart_ablation_case300_chunk",
        ),
    }

    if bus in special_dirs:

        pattern = os.path.join(
            special_dirs[bus],
            f"warmstart_ablation_raw_case{bus}_chunk*.csv",
        )

        files = sorted(glob.glob(pattern))

        if files:
            print(
                f"Ablation chunk directory: {special_dirs[bus]}"
            )
            return files, True

    # ========================================================
    # 2. Chunk files directly inside result/
    # ========================================================

    pattern = os.path.join(
        RESULT_DIR,
        f"warmstart_ablation_raw_case{bus}_chunk*.csv",
    )

    files = sorted(glob.glob(pattern))

    if files:
        return files, True

    # ========================================================
    # 3. Non-chunked file
    # ========================================================

    full_file = os.path.join(
        RESULT_DIR,
        f"warmstart_ablation_raw_case{bus}.csv",
    )

    if os.path.exists(full_file):
        return [full_file], False

    return [], False

def normalize_columns(df):
    """
    Normalize column names between different versions of the
    warm-start evaluators.
    """

    rename_map = {}

    # Different evaluator versions may use either name.
    if (
        "Speedup_vs_Cold_IPOPT" in df.columns
        and
        "Speedup" not in df.columns
    ):
        rename_map[
            "Speedup_vs_Cold_IPOPT"
        ] = "Speedup"

    if (
        "NN_Initialized_IPOPT_Time_s" in df.columns
        and
        "IPOPT_Time_s" not in df.columns
    ):
        rename_map[
            "NN_Initialized_IPOPT_Time_s"
        ] = "IPOPT_Time_s"

    df = df.rename(
        columns=rename_map
    )

    return df


def validate_required_columns(df, bus):
    """Check that required raw-result columns exist."""

    required = {
        "Architecture",
        "Run",
        "Instance_ID",
        "Success",
        "NN_Time_s",
        "Total_Online_Time_s",
        "Cold_IPOPT_Time_s",
        "Feasible_Obj_Gap_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{bus}-bus data are missing required "
            f"columns: {sorted(missing)}"
        )


def convert_success_column(df):
    """Robustly convert Success to Boolean."""

    if df["Success"].dtype == bool:
        return df

    df["Success"] = (
        df["Success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return df


def ci95_from_runs(values):
    """
    95% confidence interval of the mean using the independent
    run-level observations.

    With five independent runs:
        df = 4
    """

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan, np.nan

    mean = np.mean(x)

    if n == 1:
        return mean, mean

    sd = np.std(
        x,
        ddof=1,
    )

    sem = sd / np.sqrt(n)

    critical = t.ppf(
        0.975,
        df=n - 1,
    )

    margin = critical * sem

    return (
        mean - margin,
        mean + margin,
    )


# ============================================================
# Load one grid case
# ============================================================

def load_case(bus):

    print("\n" + "=" * 78)
    print(f"LOADING {bus}-BUS WARM-START RESULTS")
    print("=" * 78)

    # --------------------------------------------------------
    # Main architectures
    # --------------------------------------------------------

    main_files, main_chunked = (
        find_main_raw_files(bus)
    )

    if not main_files:
        raise FileNotFoundError(
            f"No main warm-start raw files found "
            f"for case {bus}."
        )

    print(
        f"Main files     : {len(main_files)} "
        f"({'chunked' if main_chunked else 'non-chunked'})"
    )

    main_df = read_csvs(
        main_files
    )

    # --------------------------------------------------------
    # Ablated PINN
    # --------------------------------------------------------

    abl_files, abl_chunked = (
        find_ablation_raw_files(bus)
    )

    if abl_files:

        print(
            f"Ablation files : {len(abl_files)} "
            f"({'chunked' if abl_chunked else 'non-chunked'})"
        )

        abl_df = read_csvs(
            abl_files
        )

        df = pd.concat(
            [
                main_df,
                abl_df,
            ],
            ignore_index=True,
        )

    else:

        print(
            "Ablation files : NONE FOUND"
        )

        df = main_df.copy()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df = normalize_columns(
        df
    )

    validate_required_columns(
        df,
        bus,
    )

    df = convert_success_column(
        df
    )

    df["Run"] = (
        df["Run"]
        .astype(int)
    )

    df["Instance_ID"] = (
        df["Instance_ID"]
        .astype(int)
    )

    df["Bus"] = bus

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    key_cols = [
        "Architecture",
        "Run",
        "Instance_ID",
    ]

    duplicate_mask = (
        df.duplicated(
            subset=key_cols,
            keep=False,
        )
    )

    n_duplicate_rows = int(
        duplicate_mask.sum()
    )

    if n_duplicate_rows > 0:

        print(
            "\nERROR: Duplicate architecture/run/instance "
            f"rows detected: {n_duplicate_rows}"
        )

        duplicate_df = (
            df.loc[
                duplicate_mask,
                key_cols + ["_source_file"],
            ]
            .sort_values(key_cols)
        )

        duplicate_file = os.path.join(
            RESULT_DIR,
            f"warmstart_duplicates_case{bus}.csv",
        )

        duplicate_df.to_csv(
            duplicate_file,
            index=False,
        )

        raise RuntimeError(
            f"Duplicate rows detected for case {bus}. "
            f"See {duplicate_file}"
        )

    # --------------------------------------------------------
    # Completeness diagnostics
    # --------------------------------------------------------

    print(
        "\nArchitectures:"
    )

    architectures = sorted(
        df["Architecture"]
        .dropna()
        .unique()
    )

    for arch in architectures:

        arch_df = df[
            df["Architecture"] == arch
        ]

        runs = sorted(
            arch_df["Run"]
            .unique()
        )

        n_instances = (
            arch_df["Instance_ID"]
            .nunique()
        )

        print(
            f"  {arch:<20s} "
            f"runs={runs} "
            f"unique_instances={n_instances} "
            f"rows={len(arch_df)}"
        )

        if len(runs) != EXPECTED_RUNS:

            print(
                f"    WARNING: expected "
                f"{EXPECTED_RUNS} runs."
            )

        # Per-run counts
        counts = (
            arch_df.groupby("Run")
            ["Instance_ID"]
            .nunique()
        )

        for run, count in counts.items():

            if count != n_instances:

                print(
                    f"    WARNING: run {run} has "
                    f"{count} instances while architecture "
                    f"union contains {n_instances}."
                )

    # --------------------------------------------------------
    # Save combined raw case
    # --------------------------------------------------------

    combined_out = os.path.join(
        RESULT_DIR,
        f"warmstart_all_raw_case{bus}.csv",
    )

    df.drop(
        columns=["_source_file"],
        errors="ignore",
    ).to_csv(
        combined_out,
        index=False,
    )

    print(
        f"\nCombined raw -> {combined_out}"
    )

    return df


# ============================================================
# Run-level aggregation
# ============================================================

def calculate_runlevel(df, bus):

    rows = []

    for (arch, run), g in df.groupby(
        [
            "Architecture",
            "Run",
        ]
    ):

        success_mask = (
            g["Success"]
            .to_numpy(dtype=bool)
        )

        n = len(g)

        n_success = int(
            success_mask.sum()
        )

        success_rate = (
            n_success / n
            if n > 0
            else np.nan
        )

        # --------------------------------------------
        # Runtime
        #
        # Runtime exists even if restoration failed,
        # so average it over attempted instances.
        # --------------------------------------------

        nn_time = (
            g["NN_Time_s"]
            .astype(float)
            .mean()
        )

        total_online_time = (
            g["Total_Online_Time_s"]
            .astype(float)
            .mean()
        )

        cold_time = (
            g["Cold_IPOPT_Time_s"]
            .astype(float)
            .mean()
        )

        # --------------------------------------------
        # Speedup
        #
        # Prefer recomputing from mean times:
        #
        #   mean cold time / mean online time
        #
        # rather than averaging per-instance ratios.
        # --------------------------------------------

        speedup = (
            cold_time / total_online_time
            if (
                np.isfinite(total_online_time)
                and total_online_time > 0
            )
            else np.nan
        )

        # --------------------------------------------
        # Feasible objective deviation
        #
        # Only successful restored solutions are
        # meaningful here.
        # --------------------------------------------

        successful = g[
            g["Success"]
        ]

        if len(successful) > 0:

            feasible_gap = (
                successful[
                    "Feasible_Obj_Gap_pct"
                ]
                .astype(float)
                .mean()
            )

        else:

            feasible_gap = np.nan

        # --------------------------------------------
        # Optional IPOPT restoration time
        # --------------------------------------------

        if "IPOPT_Time_s" in g.columns:

            ipopt_time = (
                g["IPOPT_Time_s"]
                .astype(float)
                .mean()
            )

        else:

            ipopt_time = (
                total_online_time
                -
                nn_time
            )

        rows.append({
            "Bus":
                bus,

            "Architecture":
                arch,

            "Run":
                int(run),

            "N_Attempted":
                n,

            "N_Success":
                n_success,

            "Success_Rate":
                success_rate,

            "NN_Time_s":
                nn_time,

            "IPOPT_Restoration_Time_s":
                ipopt_time,

            "Total_Online_Time_s":
                total_online_time,

            "Cold_IPOPT_Time_s":
                cold_time,

            "Speedup":
                speedup,

            "Feasible_Obj_Gap_pct":
                feasible_gap,
        })

    run_df = pd.DataFrame(
        rows
    )

    run_df = run_df.sort_values(
        [
            "Architecture",
            "Run",
        ]
    ).reset_index(
        drop=True
    )

    run_out = os.path.join(
        RESULT_DIR,
        f"warmstart_final_runlevel_case{bus}.csv",
    )

    run_df.to_csv(
        run_out,
        index=False,
    )

    print(
        f"Run-level     -> {run_out}"
    )

    return run_df


# ============================================================
# Across-run summary
# ============================================================

def summarize_across_runs(run_df, bus):

    metrics = [
        "Success_Rate",
        "NN_Time_s",
        "IPOPT_Restoration_Time_s",
        "Total_Online_Time_s",
        "Cold_IPOPT_Time_s",
        "Speedup",
        "Feasible_Obj_Gap_pct",
    ]

    rows = []

    for arch, g in run_df.groupby(
        "Architecture"
    ):

        row = {
            "Bus": bus,
            "Architecture": arch,
            "N_Runs": g["Run"].nunique(),
        }

        for metric in metrics:

            values = (
                g[metric]
                .to_numpy(
                    dtype=float
                )
            )

            finite = values[
                np.isfinite(values)
            ]

            if len(finite) == 0:

                mean = np.nan
                sd = np.nan
                ci_low = np.nan
                ci_high = np.nan

            else:

                mean = float(
                    np.mean(finite)
                )

                sd = (
                    float(
                        np.std(
                            finite,
                            ddof=1,
                        )
                    )
                    if len(finite) > 1
                    else 0.0
                )

                ci_low, ci_high = (
                    ci95_from_runs(
                        finite
                    )
                )

            row[
                f"{metric}_Mean"
            ] = mean

            row[
                f"{metric}_SD"
            ] = sd

            row[
                f"{metric}_CI95_Low"
            ] = ci_low

            row[
                f"{metric}_CI95_High"
            ] = ci_high

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "Architecture"
    ).reset_index(
        drop=True
    )

    summary_out = os.path.join(
        RESULT_DIR,
        f"warmstart_final_summary_case{bus}.csv",
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    print(
        f"Summary       -> {summary_out}"
    )

    return summary


# ============================================================
# Paper-friendly formatting
# ============================================================

def format_mean_sd(mean, sd, decimals=4):

    if not np.isfinite(mean):
        return "NA"

    if not np.isfinite(sd):
        return f"{mean:.{decimals}f}"

    return (
        f"{mean:.{decimals}f}"
        f" +/- "
        f"{sd:.{decimals}f}"
    )


def build_paper_table(all_summary):

    rows = []

    for _, r in all_summary.iterrows():

        success_mean = (
            100.0
            * r[
                "Success_Rate_Mean"
            ]
        )

        success_sd = (
            100.0
            * r[
                "Success_Rate_SD"
            ]
        )

        rows.append({
            "Case":
                int(
                    r["Bus"]
                ),

            "Architecture":
                r[
                    "Architecture"
                ],

            "Success (%)":
                format_mean_sd(
                    success_mean,
                    success_sd,
                    decimals=2,
                ),

            "Online Time (s)":
                format_mean_sd(
                    r[
                        "Total_Online_Time_s_Mean"
                    ],
                    r[
                        "Total_Online_Time_s_SD"
                    ],
                    decimals=6,
                ),

            "Speedup":
                format_mean_sd(
                    r[
                        "Speedup_Mean"
                    ],
                    r[
                        "Speedup_SD"
                    ],
                    decimals=3,
                ),

            "Feasible Obj. Dev. (%)":
                format_mean_sd(
                    r[
                        "Feasible_Obj_Gap_pct_Mean"
                    ],
                    r[
                        "Feasible_Obj_Gap_pct_SD"
                    ],
                    decimals=6,
                ),
        })

    paper_df = pd.DataFrame(
        rows
    )

    out = os.path.join(
        RESULT_DIR,
        "warmstart_paper_table.csv",
    )

    paper_df.to_csv(
        out,
        index=False,
    )

    return paper_df, out


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Summarize feasibility-restored "
            "ACOPF warm-start results."
        )
    )

    parser.add_argument(
        "--buses",
        nargs="+",
        type=int,
        default=BUSES,
        help=(
            "Grid cases to process. "
            "Default: 14 57 162 300"
        ),
    )

    args = parser.parse_args()

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    all_runlevel = []
    all_summary = []

    for bus in args.buses:

        try:

            raw_df = load_case(
                bus
            )

            run_df = calculate_runlevel(
                raw_df,
                bus,
            )

            summary_df = summarize_across_runs(
                run_df,
                bus,
            )

            all_runlevel.append(
                run_df
            )

            all_summary.append(
                summary_df
            )

        except Exception as exc:

            print(
                f"\nERROR while processing "
                f"{bus}-bus case:"
            )

            print(
                str(exc)
            )

            sys.exit(1)

    # ========================================================
    # Combined outputs across all cases
    # ========================================================

    all_runlevel_df = pd.concat(
        all_runlevel,
        ignore_index=True,
    )

    all_summary_df = pd.concat(
        all_summary,
        ignore_index=True,
    )

    all_runlevel_out = os.path.join(
        RESULT_DIR,
        "warmstart_final_runlevel_all_cases.csv",
    )

    all_summary_out = os.path.join(
        RESULT_DIR,
        "warmstart_final_summary_all_cases.csv",
    )

    all_runlevel_df.to_csv(
        all_runlevel_out,
        index=False,
    )

    all_summary_df.to_csv(
        all_summary_out,
        index=False,
    )

    # ========================================================
    # Paper-ready table
    # ========================================================

    paper_df, paper_out = (
        build_paper_table(
            all_summary_df
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FEASIBILITY-RESTORED ACOPF PERFORMANCE"
    )

    print(
        "=" * 100
    )

    print(
        paper_df.to_string(
            index=False
        )
    )

    print(
        "\nFinal files:"
    )

    print(
        f"  {all_runlevel_out}"
    )

    print(
        f"  {all_summary_out}"
    )

    print(
        f"  {paper_out}"
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()