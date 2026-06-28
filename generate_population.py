"""
Module 1-2: Population Generator + Hotel & Stop Assignment

Pure function interface:
    generate_population(cfg, rng) -> (df, assumptions)
    assign_hotels_stops(df, cfg, rng, assumptions) -> (df, assumptions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config dataclasses (populated by run_all.py from user inputs)
# ---------------------------------------------------------------------------

@dataclass
class PopulationConfig:
    # One of these is provided; the other is derived.
    total_delegates: int | None = None
    total_groups: int | None = None

    # {size: fraction} e.g. {1: 0.05, 2: 0.47, ...}; must sum to 1.0 after rescaling
    group_size_distribution: dict[int, float] = field(default_factory=dict)

    # {nights: fraction} derived from occupancy curve
    nights_distribution: dict[int, float] = field(default_factory=dict)

    # Core-stay window (used to block scheduling days)
    core_stay_start: date | None = None
    core_stay_end: date | None = None

    # Overall window
    window_start: date | None = None
    window_end: date | None = None

    # Check-in/out policy:
    #   "arrival_based" — sample check-in from checkin_distribution, check-out = check-in + nights
    #   "randomized"    — sample check-out from checkout_distribution, check-in = check-out - nights
    #   "fixed"         — everyone checks out on checkout_fixed_date
    checkout_policy: str = "arrival_based"
    checkin_distribution: dict[date, float] = field(default_factory=dict)   # for arrival_based
    checkout_fixed_date: date | None = None                                   # for fixed
    checkout_distribution: dict[date, float] = field(default_factory=dict)   # for randomized

    random_seed: int = 42


@dataclass
class HotelConfig:
    # [{name, capacity}]
    hotels: list[dict[str, Any]] = field(default_factory=list)
    # {hotel_name: stop_name}
    hotel_to_stop: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rescale_dist(d: dict, label: str, assumptions: list[str]) -> dict:
    """Rescale a {key: fraction} dict to sum exactly 1.0, logging the action."""
    total = sum(d.values())
    if abs(total - 1.0) > 1e-9:
        factor = 1.0 / total
        d = {k: v * factor for k, v in d.items()}
        assumptions.append(
            f"RESCALE: {label} percentages summed to {total:.6f} (not 1.0). "
            f"Each fraction multiplied by {factor:.6f} to normalise."
        )
    return d


def _largest_remainder(keys: list, fractions: list[float], total: int) -> list[int]:
    """Allocate `total` integers across `keys` using the largest-remainder method."""
    exact = [f * total for f in fractions]
    floors = [int(x) for x in exact]
    remainders = [(exact[i] - floors[i], i) for i in range(len(exact))]
    remainder_slots = total - sum(floors)
    remainders.sort(key=lambda x: -x[0])
    for i in range(remainder_slots):
        floors[remainders[i][1]] += 1
    return floors


# ---------------------------------------------------------------------------
# Module 1: Population Generator
# ---------------------------------------------------------------------------

def generate_population(cfg: PopulationConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    assumptions: list[str] = []

    # 1. Resolve delegate vs group count
    gs_dist = _rescale_dist(dict(cfg.group_size_distribution), "group-size distribution", assumptions)
    sizes = list(gs_dist.keys())
    fracs = [gs_dist[s] for s in sizes]
    expected_group_size = sum(s * f for s, f in zip(sizes, fracs))

    if cfg.total_groups is not None:
        n_groups = cfg.total_groups
        assumptions.append(f"GROUP COUNT: User provided {n_groups} groups directly.")
    elif cfg.total_delegates is not None:
        n_groups = round(cfg.total_delegates / expected_group_size)
        assumptions.append(
            f"GROUP COUNT: Derived {n_groups} groups from {cfg.total_delegates} delegates "
            f"÷ expected group size {expected_group_size:.3f}."
        )
    else:
        raise ValueError("PopulationConfig must specify either total_delegates or total_groups.")

    # 2. Allocate group counts per size bucket (largest-remainder)
    counts_per_size = _largest_remainder(sizes, fracs, n_groups)

    # 3. Build raw group list
    group_sizes: list[int] = []
    for s, cnt in zip(sizes, counts_per_size):
        group_sizes.extend([s] * cnt)
    rng.shuffle(group_sizes)

    # 4. Nudge to hit exact delegate target (if delegate-driven)
    if cfg.total_delegates is not None:
        current_delegates = sum(group_sizes)
        delta = cfg.total_delegates - current_delegates
        if delta != 0:
            direction = 1 if delta > 0 else -1
            nudged = 0
            for i in rng.permutation(n_groups):
                if delta == 0:
                    break
                new_size = group_sizes[i] + direction
                if new_size >= 1:
                    group_sizes[i] = new_size
                    delta -= direction
                    nudged += 1
            assumptions.append(
                f"NUDGE: {nudged} group sizes adjusted ±1 to match exact delegate target "
                f"{cfg.total_delegates}."
            )

    # 5. Assign nights per group (largest-remainder)
    nights_dist = _rescale_dist(dict(cfg.nights_distribution), "nights distribution", assumptions)
    night_values = list(nights_dist.keys())
    night_fracs = [nights_dist[n] for n in night_values]
    night_counts = _largest_remainder(night_values, night_fracs, n_groups)
    nights_list: list[int] = []
    for nv, nc in zip(night_values, night_counts):
        nights_list.extend([nv] * nc)
    rng.shuffle(nights_list)

    # 6. Assign check-in / check-out dates
    if cfg.checkout_policy == "arrival_based":
        # Sample check-in first from arrival distribution, then check-out = check-in + nights
        ci_dist = _rescale_dist(dict(cfg.checkin_distribution), "check-in (arrival) distribution", assumptions)
        ci_dates = list(ci_dist.keys())
        ci_fracs = [ci_dist[d] for d in ci_dates]
        ci_counts = _largest_remainder(ci_dates, ci_fracs, n_groups)
        raw_ci: list[date] = []
        for d, cnt in zip(ci_dates, ci_counts):
            raw_ci.extend([d] * cnt)
        rng.shuffle(raw_ci)
        checkin_dates = raw_ci
        assumptions.append("CHECKOUT POLICY: Arrival-based — check-in sampled from day-of-week distribution; check-out = check-in + nights.")

    elif cfg.checkout_policy == "fixed":
        if cfg.checkout_fixed_date is None:
            raise ValueError("checkout_fixed_date must be set when checkout_policy='fixed'.")
        # Everyone checks out on the same date; check-in = fixed_date - nights
        assumptions.append(f"CHECKOUT POLICY: Fixed — all groups check out {cfg.checkout_fixed_date}.")
        checkin_dates = None  # handled below

    elif cfg.checkout_policy == "randomized":
        co_dist = _rescale_dist(dict(cfg.checkout_distribution), "checkout distribution", assumptions)
        co_dates_keys = list(co_dist.keys())
        co_fracs = [co_dist[d] for d in co_dates_keys]
        co_counts = _largest_remainder(co_dates_keys, co_fracs, n_groups)
        raw_co: list[date] = []
        for d, cnt in zip(co_dates_keys, co_counts):
            raw_co.extend([d] * cnt)
        rng.shuffle(raw_co)
        assumptions.append("CHECKOUT POLICY: Randomized — check-out sampled from distribution; check-in = check-out − nights.")
        checkin_dates = None  # handled below
    else:
        raise ValueError(f"Unknown checkout_policy: {cfg.checkout_policy!r}")

    # 7. Build rows
    rows = []
    clamped_count = 0
    core_adjusted_count = 0
    for i in range(n_groups):
        nights_nominal = nights_list[i]

        if cfg.checkout_policy == "arrival_based":
            ci = checkin_dates[i]
            # Ensure check-in is early enough that checkout covers the full core stay
            if cfg.core_stay_end:
                latest_valid_ci = cfg.core_stay_end - timedelta(days=nights_nominal)
                if ci > latest_valid_ci:
                    ci = max(latest_valid_ci, cfg.window_start or latest_valid_ci)
                    core_adjusted_count += 1
            co_nominal = ci + timedelta(days=nights_nominal)
            co = min(co_nominal, cfg.window_end) if cfg.window_end else co_nominal
            if co != co_nominal:
                clamped_count += 1

        elif cfg.checkout_policy == "fixed":
            co = cfg.checkout_fixed_date
            ci_nominal = co - timedelta(days=nights_nominal)
            ci = max(ci_nominal, cfg.window_start) if cfg.window_start else ci_nominal
            if ci != ci_nominal:
                clamped_count += 1

        else:  # randomized
            co = raw_co[i]
            ci_nominal = co - timedelta(days=nights_nominal)
            ci = max(ci_nominal, cfg.window_start) if cfg.window_start else ci_nominal
            if ci != ci_nominal:
                clamped_count += 1

        nights_realized = (co - ci).days
        rows.append({
            "GroupID": f"G{i+1:05d}",
            "GroupSize": group_sizes[i],
            "NightsNominal": nights_nominal,
            "CheckIn": ci,
            "CheckOut": co,
            "NightsRealized": nights_realized,
        })

    if clamped_count:
        assumptions.append(
            f"CLAMPING: {clamped_count} groups had checkout clamped to fit within the stay window."
        )
    if core_adjusted_count:
        assumptions.append(
            f"CORE COVERAGE: {core_adjusted_count} groups had their check-in shifted earlier "
            f"to ensure their stay covers the mandatory core period (through {cfg.core_stay_end})."
        )

    df = pd.DataFrame(rows)

    total_delegates_out = int(df["GroupSize"].sum())
    assumptions.append(f"POPULATION OUTPUT: {n_groups} groups, {total_delegates_out} total delegates.")

    return df, assumptions


# ---------------------------------------------------------------------------
# Module 2: Hotel & Stop Assignment
# ---------------------------------------------------------------------------

def assign_hotels_stops(
    df: pd.DataFrame,
    cfg: HotelConfig,
    rng: np.random.Generator,
    assumptions: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    if not cfg.hotels:
        assumptions.append("WARNING: No hotels provided — Hotel and TransportStop columns left blank.")
        df["Hotel"] = None
        df["TransportStop"] = None
        return df, assumptions

    hotel_names = [h["name"] for h in cfg.hotels]
    capacities = np.array([h["capacity"] for h in cfg.hotels], dtype=float)
    weights = capacities / capacities.sum()

    df["Hotel"] = rng.choice(hotel_names, size=len(df), p=weights)

    # Build hotel→stop mapping with mismatch detection
    unresolved: list[str] = []
    hotel_to_stop: dict[str, str] = {}
    for hotel_name in hotel_names:
        stop = cfg.hotel_to_stop.get(hotel_name)
        if stop is None:
            unresolved.append(hotel_name)
            hotel_to_stop[hotel_name] = "NOT AVAILABLE"
        else:
            hotel_to_stop[hotel_name] = stop

    if unresolved:
        assumptions.append(
            f"UNRESOLVED HOTEL→STOP MAPPINGS ({len(unresolved)}): "
            + ", ".join(f"'{h}'" for h in unresolved)
            + " — no matching stop found. Marked NOT AVAILABLE in output. "
            "Please confirm correct stop assignment for these hotels."
        )

    # Also flag any hotel names in cfg.hotel_to_stop that don't appear in the hotel list
    orphan_mappings = [h for h in cfg.hotel_to_stop if h not in hotel_names]
    if orphan_mappings:
        assumptions.append(
            f"ORPHAN STOP MAPPINGS ({len(orphan_mappings)}): "
            + ", ".join(f"'{h}'" for h in orphan_mappings)
            + " — referenced in stop map but not in hotel inventory. "
            "Check for name mismatches (e.g. 'HGI DTC' vs 'HGI TC')."
        )

    df["TransportStop"] = df["Hotel"].map(hotel_to_stop)

    assumptions.append(
        f"HOTEL ASSIGNMENT: Weighted by room capacity. "
        f"Hotels: {', '.join(hotel_names)}."
    )

    return df, assumptions
