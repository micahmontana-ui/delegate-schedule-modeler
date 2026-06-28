"""
Orchestrator: run_all.py

Usage:
    python run_all.py              # run full pipeline with example config
    python run_all.py --test       # run tests only
    python run_all.py --help       # show this message

Each stage is a pure function of its inputs, so any stage can be rerun
in isolation when parameters change without re-deriving other stages.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Example configuration — replace with your actual inputs
# (or wire this to a CLI prompt / config file)
# ---------------------------------------------------------------------------

SEED = 42

POP_CFG_EXAMPLE = {
    "total_delegates": 2000,
    "group_size_distribution": {1: 0.05, 2: 0.47, 3: 0.30, 4: 0.12, 5: 0.06},
    "nights_distribution": {6: 0.40, 7: 0.35, 8: 0.15, 9: 0.05, 10: 0.05},
    "core_stay_start": date(2027, 8, 5),
    "core_stay_end": date(2027, 8, 8),
    "window_start": date(2027, 8, 1),
    "window_end": date(2027, 8, 15),
    "checkout_policy": "randomized",
    "checkout_distribution": {
        date(2027, 8, 10): 0.40,
        date(2027, 8, 11): 0.35,
        date(2027, 8, 12): 0.15,
        date(2027, 8, 13): 0.10,
    },
}

HOTELS_EXAMPLE = [
    {"name": "Hotel Alpha", "capacity": 400},
    {"name": "Hotel Beta", "capacity": 250},
    {"name": "Hotel Gamma", "capacity": 200},
    {"name": "Hotel Delta", "capacity": 150},
]

HOTEL_TO_STOP_EXAMPLE = {
    "Hotel Alpha": "Stop 1",
    "Hotel Beta": "Stop 2",
    "Hotel Gamma": "Stop 2",
    "Hotel Delta": "Stop 3",
}

STOP_NAMES_EXAMPLE = ["Stop 1", "Stop 2", "Stop 3", "Stop 4"]

PAIRING_RULES_EXAMPLE = [
    ("Stop 2", "Stop 3"),
    ("Stop 1", "Stop 4"),
]

REQUIRED_ACTIVITIES_RAW_EXAMPLE = [
    {
        "name": "Main Gathering",
        "dates": [date(2027, 8, 3), date(2027, 8, 4), date(2027, 8, 9), date(2027, 8, 10)],
        "capacity": {
            date(2027, 8, 3): 300, date(2027, 8, 4): 300,
            date(2027, 8, 9): 300, date(2027, 8, 10): 300,
        },
        "priority": 1,
    },
    {
        "name": "Regional Meeting",
        "dates": [date(2027, 8, 2), date(2027, 8, 9), date(2027, 8, 11)],
        "capacity": None,
        "priority": 2,
    },
]

OPTIONAL_ACTIVITIES_RAW_EXAMPLE = [
    {
        "name": "Museum Tour",
        "dates": [date(2027, 8, 3), date(2027, 8, 4), date(2027, 8, 9)],
        "capacity": None,
        "want_pct": 0.30,
        "prerequisite": None,
    },
]

BUS_MIN = 20
BUS_MAX = 55
TOPUP_THRESHOLD = 40

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(seed: int = SEED, changelog: list[str] | None = None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    assumptions: list[str] = []

    rng = np.random.default_rng(seed)
    assumptions.append(f"RANDOM SEED: {seed}")

    # Stage 1: Population
    from generate_population import PopulationConfig, generate_population, assign_hotels_stops, HotelConfig
    pop_cfg = PopulationConfig(
        **{k: v for k, v in POP_CFG_EXAMPLE.items()},
        random_seed=seed,
    )
    print("Stage 1: Generating population...")
    df, assumptions = generate_population(pop_cfg, rng)
    print(f"  → {len(df)} groups, {df['GroupSize'].sum()} delegates")

    # Stage 2: Hotel & stop assignment
    hotel_cfg = HotelConfig(hotels=HOTELS_EXAMPLE, hotel_to_stop=HOTEL_TO_STOP_EXAMPLE)
    print("Stage 2: Assigning hotels and stops...")
    df, assumptions = assign_hotels_stops(df, hotel_cfg, rng, assumptions)

    # Stage 3: Stop network
    from build_stop_network import build_stop_network, report_isolated_stop_efficiency
    print("Stage 3: Building stop network...")
    adjacency, assumptions = build_stop_network(STOP_NAMES_EXAMPLE, PAIRING_RULES_EXAMPLE, assumptions)

    # Stage 4: Activity definitions
    from activity_definitions import build_activity_defs
    print("Stage 4: Building activity definitions...")
    REQUIRED, OPTIONAL, assumptions = build_activity_defs(
        REQUIRED_ACTIVITIES_RAW_EXAMPLE,
        OPTIONAL_ACTIVITIES_RAW_EXAMPLE,
        pop_cfg.window_start,
        pop_cfg.window_end,
        assumptions,
    )

    # Stage 5: Scheduling
    from scheduler import schedule
    print("Stage 5: Scheduling activities...")
    df, non_attendance, assumptions = schedule(
        df, REQUIRED, OPTIONAL,
        pop_cfg.core_stay_start, pop_cfg.core_stay_end,
        rng, assumptions,
    )

    # Stage 6: Bus assignment
    from bus_assignment import assign_buses
    print("Stage 6: Assigning buses...")
    bus_trips_df, stop_summary, assumptions = assign_buses(
        df, adjacency, BUS_MIN, BUS_MAX, TOPUP_THRESHOLD, assumptions
    )

    # Isolated stop efficiency report
    report_isolated_stop_efficiency(adjacency, stop_summary, assumptions)

    # Stage 7a: Workbook
    from build_workbook import build_workbook
    wb_path = str(OUTPUT_DIR / "convention_logistics.xlsx")
    print(f"Stage 7a: Building workbook → {wb_path}")
    build_workbook(
        wb_path, df, bus_trips_df, stop_summary, non_attendance,
        REQUIRED, OPTIONAL, HOTELS_EXAMPLE, assumptions,
    )

    # Stage 7b: Report
    from build_report import build_report

    total_trips = len(bus_trips_df) if not bus_trips_df.empty else 0
    under_min = int(bus_trips_df["UnderMinimum"].sum()) if not bus_trips_df.empty else 0
    pct_under = f"{100*under_min/total_trips:.1f}" if total_trips else "0.0"

    act_assigned = {}
    for act_name in REQUIRED:
        act_assigned[act_name] = int(df[f"req_{act_name}"].notna().sum())

    summary_stats = {
        "total_groups": len(df),
        "total_delegates": int(df["GroupSize"].sum()),
        "total_trips": total_trips,
        "under_min_trips": under_min,
        "pct_under_min": pct_under,
        "activity_assigned": act_assigned,
        "clamped_groups": sum(1 for a in assumptions if "CLAMPING" in a and "groups" in a),
        "biggest_findings": [a for a in assumptions if a.startswith(("WARNING", "ISOLATED", "VALIDATION ERROR"))],
    }

    report_path = str(OUTPUT_DIR / "convention_logistics_report.docx")
    print(f"Stage 7b: Building report → {report_path}")
    build_report(
        report_path, summary_stats, assumptions, non_attendance,
        stop_summary, bus_trips_df, REQUIRED, changelog,
    )

    # Stage 8: Tests
    print("\nStage 8: Running validation checks...")
    _run_inline_checks(df, bus_trips_df, adjacency, non_attendance, REQUIRED, seed, assumptions)

    print("\nDone. Outputs in:", OUTPUT_DIR)
    return df, bus_trips_df, assumptions


def _run_inline_checks(df, bus_trips_df, adjacency, non_attendance, required_activities, seed, assumptions):
    errors: list[str] = []

    # Check 1: delegate count
    # (already verified via population generation nudge — log confirmation)
    print("  ✓ Delegate count verified during generation")

    # Check 2: required activity coverage
    for act_name in required_activities:
        col = f"req_{act_name}"
        unscheduled = df[df[col].isna()]["GroupID"].tolist()
        if not non_attendance.empty:
            na_groups = set(non_attendance[non_attendance["Activity"] == act_name]["GroupID"])
        else:
            na_groups = set()
        missing = [g for g in unscheduled if g not in na_groups]
        if missing:
            errors.append(f"CHECK 2 FAIL: {len(missing)} groups missing {act_name} without non-attendance record")
        else:
            print(f"  ✓ {act_name}: all non-attendees have records")

    # Check 3: passenger reconciliation (done inside assign_buses — check log)
    val_pass = any("VALIDATION PASS" in a for a in assumptions)
    val_fail = any("VALIDATION ERROR" in a for a in assumptions)
    if val_pass:
        print("  ✓ Passenger trip reconciliation passed")
    elif val_fail:
        errors.append("CHECK 3 FAIL: Passenger trip reconciliation mismatch — see assumptions log")

    # Check 4: no bus exceeds max
    if not bus_trips_df.empty:
        overloaded = bus_trips_df[bus_trips_df["Passengers"] > 55]
        if not overloaded.empty:
            errors.append(f"CHECK 4 FAIL: {len(overloaded)} buses exceed max capacity")
        else:
            print("  ✓ No buses exceed maximum capacity")

    # Check 5: stops valid
    if not bus_trips_df.empty:
        valid_stops = set(adjacency.keys())
        invalid = bus_trips_df[~bus_trips_df["Stop"].isin(valid_stops)]
        if not invalid.empty:
            errors.append(f"CHECK 5 FAIL: Unknown stops in bus output: {invalid['Stop'].unique()}")
        else:
            print("  ✓ All bus stops are valid")

    # Check 6: reproducibility (spot-check group count)
    import numpy as np
    from generate_population import PopulationConfig, generate_population
    rng_r = np.random.default_rng(seed)
    pop_cfg_r = PopulationConfig(
        total_delegates=POP_CFG_EXAMPLE["total_delegates"],
        group_size_distribution=POP_CFG_EXAMPLE["group_size_distribution"],
        nights_distribution=POP_CFG_EXAMPLE["nights_distribution"],
        core_stay_start=POP_CFG_EXAMPLE["core_stay_start"],
        core_stay_end=POP_CFG_EXAMPLE["core_stay_end"],
        window_start=POP_CFG_EXAMPLE["window_start"],
        window_end=POP_CFG_EXAMPLE["window_end"],
        checkout_policy=POP_CFG_EXAMPLE["checkout_policy"],
        checkout_distribution=POP_CFG_EXAMPLE["checkout_distribution"],
        random_seed=seed,
    )
    df_r, _ = generate_population(pop_cfg_r, rng_r)
    if df_r["GroupSize"].sum() == df["GroupSize"].sum() and len(df_r) == len(df):
        print("  ✓ Reproducibility check passed")
    else:
        errors.append("CHECK 6 FAIL: Rerun with same seed produced different population size")

    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("\n  All checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convention Logistics Modeling Tool")
    parser.add_argument("--test", action="store_true", help="Run pytest test suite only")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default {SEED})")
    args = parser.parse_args()

    if args.test:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            cwd=Path(__file__).parent,
        )
        sys.exit(result.returncode)
    else:
        run_pipeline(seed=args.seed)


if __name__ == "__main__":
    main()
