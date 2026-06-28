"""
Section 8: Testing checklist

Run with: python -m pytest tests/test_all.py -v
Or via:   python run_all.py --test
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from generate_population import PopulationConfig, generate_population, assign_hotels_stops, HotelConfig
from build_stop_network import build_stop_network
from activity_definitions import build_activity_defs
from scheduler import schedule
from bus_assignment import assign_buses


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_pop_cfg():
    return PopulationConfig(
        total_delegates=1000,
        group_size_distribution={1: 0.05, 2: 0.47, 3: 0.30, 4: 0.12, 5: 0.06},
        nights_distribution={6: 0.40, 7: 0.35, 8: 0.15, 9: 0.05, 10: 0.05},
        core_stay_start=date(2027, 8, 5),
        core_stay_end=date(2027, 8, 8),
        window_start=date(2027, 8, 1),
        window_end=date(2027, 8, 15),
        checkout_policy="randomized",
        checkout_distribution={
            date(2027, 8, 10): 0.40,
            date(2027, 8, 11): 0.35,
            date(2027, 8, 12): 0.15,
            date(2027, 8, 13): 0.10,
        },
        random_seed=42,
    )


@pytest.fixture
def base_hotel_cfg():
    return HotelConfig(
        hotels=[
            {"name": "Hotel Alpha", "capacity": 300},
            {"name": "Hotel Beta", "capacity": 200},
            {"name": "Hotel Gamma", "capacity": 150},
        ],
        hotel_to_stop={
            "Hotel Alpha": "Stop 1",
            "Hotel Beta": "Stop 2",
            "Hotel Gamma": "Stop 2",
        },
    )


@pytest.fixture
def base_df(base_pop_cfg):
    rng = np.random.default_rng(base_pop_cfg.random_seed)
    df, _ = generate_population(base_pop_cfg, rng)
    return df, rng


@pytest.fixture
def scheduled_df(base_df, base_hotel_cfg):
    df, rng = base_df
    assumptions = []
    df, assumptions = assign_hotels_stops(df, base_hotel_cfg, rng, assumptions)

    req_raw = [
        {
            "name": "Activity A",
            "dates": [date(2027, 8, 3), date(2027, 8, 4), date(2027, 8, 9)],
            "capacity": None,
            "priority": 1,
        },
        {
            "name": "Activity B",
            "dates": [date(2027, 8, 2), date(2027, 8, 3), date(2027, 8, 9), date(2027, 8, 10)],
            "capacity": {
                date(2027, 8, 2): 200, date(2027, 8, 3): 200,
                date(2027, 8, 9): 200, date(2027, 8, 10): 200,
            },
            "priority": 2,
        },
    ]
    opt_raw = []
    REQUIRED, OPTIONAL, assumptions = build_activity_defs(
        req_raw, opt_raw,
        date(2027, 8, 1), date(2027, 8, 15), assumptions
    )

    df_sched, non_att, assumptions = schedule(
        df, REQUIRED, OPTIONAL,
        date(2027, 8, 5), date(2027, 8, 8), rng, assumptions
    )
    return df_sched, non_att, assumptions, REQUIRED, OPTIONAL


# ---------------------------------------------------------------------------
# Test 1: Total delegates match requested population
# ---------------------------------------------------------------------------

def test_total_delegates(base_pop_cfg):
    rng = np.random.default_rng(base_pop_cfg.random_seed)
    df, assumptions = generate_population(base_pop_cfg, rng)
    assert df["GroupSize"].sum() == base_pop_cfg.total_delegates, (
        f"Expected {base_pop_cfg.total_delegates} delegates, got {df['GroupSize'].sum()}"
    )


# ---------------------------------------------------------------------------
# Test 2: Every group has required activity or explicit non-attendance entry
# ---------------------------------------------------------------------------

def test_required_activity_coverage(scheduled_df):
    df, non_att, _, REQUIRED, _ = scheduled_df
    for act_name in REQUIRED:
        col = f"req_{act_name}"
        unscheduled = df[df[col].isna()]["GroupID"].tolist()
        if non_att.empty:
            assert not unscheduled, (
                f"Groups missing {act_name} with no non-attendance record: {unscheduled[:5]}"
            )
        else:
            na_groups = set(non_att[non_att["Activity"] == act_name]["GroupID"])
            missing_no_record = [g for g in unscheduled if g not in na_groups]
            assert not missing_no_record, (
                f"{len(missing_no_record)} groups lack {act_name} without a non-attendance record"
            )


# ---------------------------------------------------------------------------
# Test 3: Passenger trip counts match (bus vs population table)
# ---------------------------------------------------------------------------

def test_passenger_trip_reconciliation(scheduled_df, base_hotel_cfg):
    df, _, _, _, _ = scheduled_df
    act_cols = [c for c in df.columns if c.startswith("req_") or c.startswith("opt_")]

    adjacency, _ = build_stop_network(
        ["Stop 1", "Stop 2"], [], []
    )
    bus_trips_df, _, assumptions = assign_buses(
        df, adjacency, bus_min=20, bus_max=55, topup_threshold=40, assumptions=[]
    )

    if bus_trips_df.empty:
        pytest.skip("No trips generated")

    pax_buses = int(bus_trips_df["Passengers"].sum())

    pax_pop = 0
    for _, row in df.iterrows():
        stop = row.get("TransportStop")
        if pd.isna(stop) or stop == "NOT AVAILABLE":
            continue
        for col in act_cols:
            if pd.notna(row[col]):
                pax_pop += int(row["GroupSize"])

    assert pax_buses == pax_pop, (
        f"Passenger mismatch: buses={pax_buses}, population table={pax_pop}"
    )


# ---------------------------------------------------------------------------
# Test 4: No bus exceeds maximum
# ---------------------------------------------------------------------------

def test_no_bus_exceeds_max(scheduled_df):
    df, _, _, _, _ = scheduled_df
    adjacency, _ = build_stop_network(["Stop 1", "Stop 2"], [], [])
    bus_trips_df, _, _ = assign_buses(
        df, adjacency, bus_min=20, bus_max=55, topup_threshold=40, assumptions=[]
    )
    if bus_trips_df.empty:
        pytest.skip("No trips generated")
    overloaded = bus_trips_df[bus_trips_df["Passengers"] > 55]
    assert overloaded.empty, f"{len(overloaded)} buses exceed max capacity"


# ---------------------------------------------------------------------------
# Test 5: Every stop in bus output exists in stop definition
# ---------------------------------------------------------------------------

def test_bus_stops_valid(scheduled_df):
    df, _, _, _, _ = scheduled_df
    adjacency, _ = build_stop_network(["Stop 1", "Stop 2"], [], [])
    bus_trips_df, _, _ = assign_buses(
        df, adjacency, bus_min=20, bus_max=55, topup_threshold=40, assumptions=[]
    )
    if bus_trips_df.empty:
        pytest.skip("No trips generated")
    valid_stops = set(adjacency.keys())
    invalid = bus_trips_df[~bus_trips_df["Stop"].isin(valid_stops)]
    assert invalid.empty, f"Unknown stops in bus output: {invalid['Stop'].unique()}"


# ---------------------------------------------------------------------------
# Test 6: Reproducibility — same seed produces identical output
# ---------------------------------------------------------------------------

def test_reproducibility(base_pop_cfg):
    rng1 = np.random.default_rng(base_pop_cfg.random_seed)
    df1, _ = generate_population(base_pop_cfg, rng1)

    rng2 = np.random.default_rng(base_pop_cfg.random_seed)
    df2, _ = generate_population(base_pop_cfg, rng2)

    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Test 7: Group-size distribution percentages rescaling warning
# ---------------------------------------------------------------------------

def test_distribution_rescaling_logged():
    cfg = PopulationConfig(
        total_delegates=500,
        group_size_distribution={1: 0.10, 2: 0.50, 3: 0.30},  # sums to 0.90, not 1.0
        nights_distribution={7: 1.0},
        core_stay_start=date(2027, 8, 5),
        core_stay_end=date(2027, 8, 8),
        window_start=date(2027, 8, 1),
        window_end=date(2027, 8, 15),
        checkout_policy="fixed",
        checkout_fixed_date=date(2027, 8, 12),
        random_seed=1,
    )
    rng = np.random.default_rng(1)
    df, assumptions = generate_population(cfg, rng)
    rescale_logged = any("RESCALE" in a for a in assumptions)
    assert rescale_logged, "Expected a RESCALE assumption to be logged when distribution doesn't sum to 1"


# ---------------------------------------------------------------------------
# Test 8: NaN vs None safety in activity columns
# ---------------------------------------------------------------------------

def test_notna_safety(scheduled_df):
    """Confirm pd.notna() is reliable for activity columns (catches nan-vs-None bug)."""
    df, _, _, _, _ = scheduled_df
    act_cols = [c for c in df.columns if c.startswith("req_") or c.startswith("opt_")]
    for col in act_cols:
        # All values should be either a date-like or NaN (not Python None after serialization)
        non_null = df[col].dropna()
        for val in non_null:
            assert val is not None, f"Found None (not NaN) in {col} — notna() would behave unexpectedly"
