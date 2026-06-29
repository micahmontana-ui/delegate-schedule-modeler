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

    # 6. Build arrival distribution lookup (used per-group below)
    # Keys are dates; values are raw weights (not yet normalized per-LOS).
    ci_dist_raw = _rescale_dist(dict(cfg.checkin_distribution), "check-in (arrival) distribution", assumptions)
    assumptions.append(
        "CHECKOUT POLICY: Arrival-based — check-in sampled from arrival distribution "
        "constrained to the valid window for each group's LOS; check-out = check-in + nights. "
        "Mandatory Nights 6–9 are guaranteed by construction (no post-hoc clamping)."
    )

    MIN_NIGHTS = 6

    # 7. Build rows — check-in is sampled per group from the arrival distribution
    # restricted to the valid range for that group's LOS:
    #   earliest_ci = core_stay_end − nights  (must reach Day 10 in time)
    #   latest_ci   = core_stay_start − 1     (must be present for Night 6)
    #   also bounded by window_start and (window_end − nights)
    rows = []
    fallback_count = 0

    for i in range(n_groups):
        nights = max(nights_list[i], MIN_NIGHTS)

        latest_ci = cfg.core_stay_start - timedelta(days=1)  # Day 6
        earliest_ci = cfg.core_stay_end - timedelta(days=nights)
        if cfg.window_start:
            earliest_ci = max(earliest_ci, cfg.window_start)
        if cfg.window_end:
            latest_ci = min(latest_ci, cfg.window_end - timedelta(days=nights))

        valid_weights = {d: w for d, w in ci_dist_raw.items() if earliest_ci <= d <= latest_ci}

        if valid_weights:
            total_w = sum(valid_weights.values())
            ci_days = list(valid_weights.keys())
            ci_probs = [valid_weights[d] / total_w for d in ci_days]
            ci = ci_days[rng.choice(len(ci_days), p=ci_probs)]
        else:
            # No arrival weight in valid range — fall back to latest valid day
            ci = latest_ci
            fallback_count += 1

        co = ci + timedelta(days=nights)
        nights_realized = (co - ci).days
        rows.append({
            "GroupID": f"G{i+1:05d}",
            "GroupSize": group_sizes[i],
            "NightsNominal": nights,
            "CheckIn": ci,
            "CheckOut": co,
            "NightsRealized": nights_realized,
        })

    if fallback_count:
        assumptions.append(
            f"ARRIVAL FALLBACK: {fallback_count} groups had no arrival weight in their valid "
            f"check-in window (LOS too long for early arrival days) — assigned to latest valid day."
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
