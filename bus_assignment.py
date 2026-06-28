"""
Module 6: Bus Assignment Engine

Pure function interface:
    assign_buses(df, adjacency, bus_min, bus_max, topup_threshold, assumptions)
        -> (bus_trips_df, stop_summary, assumptions)

Design decisions (see spec):
    1. Assign at STOP level (not hotel level)
    2. Pool different activity labels that share the same physical destination
    3. Bin-packing with greedy descent
    4. Multi-stop top-up for under-filled buses
    5. Validates total passenger-trips two independent ways
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def assign_buses(
    df: pd.DataFrame,
    adjacency: dict[str, list[str]],
    bus_min: int,
    bus_max: int,
    topup_threshold: int | None,
    assumptions: list[str],
) -> tuple[pd.DataFrame, dict[str, dict], list[str]]:
    """
    Parameters
    ----------
    df : scheduled population (must have TransportStop, req_*/opt_* columns, GroupSize)
    adjacency : stop adjacency dict from build_stop_network
    bus_min : minimum passengers per bus (trips below this are "under-minimum")
    bus_max : maximum passengers per bus
    topup_threshold : if a bus has fewer than this many passengers, try to combine
                      with an adjacent stop's under-filled bus. Defaults to bus_max.
    assumptions : running log

    Returns
    -------
    bus_trips_df : one row per bus trip, columns:
        [TripID, Date, Stop, ActivityLabels, BusNumber, Passengers, UnderMinimum]
    stop_summary : {stop: {total_trips, under_min_trips, total_passengers}}
    assumptions : updated log
    """
    if topup_threshold is None:
        topup_threshold = bus_max

    assumptions.append(
        f"BUS ASSIGNMENT: min={bus_min}, max={bus_max}, topup_threshold={topup_threshold}."
    )

    # ----------------------------------------------------------------
    # 1. Build pool: (stop, date, activity_label) -> list of (GroupID, GroupSize)
    # ----------------------------------------------------------------
    # Collect all scheduled activity columns
    act_cols = [c for c in df.columns if c.startswith("req_") or c.startswith("opt_")]

    # Pool by (stop, date) — merge all activity labels going to same stop same day
    # Key: (stop, date) -> [(GroupID, GroupSize, activity_label), ...]
    pool: dict[tuple[str, date], list[tuple[str, int, str]]] = {}

    for _, row in df.iterrows():
        stop = row.get("TransportStop")
        if pd.isna(stop) or stop == "NOT AVAILABLE":
            continue

        for col in act_cols:
            val = row[col]
            if pd.isna(val):
                continue
            act_label = col[4:] if col.startswith("req_") else col[4:]  # strip prefix
            act_date = val if isinstance(val, date) else pd.Timestamp(val).date()
            key = (stop, act_date)
            pool.setdefault(key, [])
            pool[key].append((row["GroupID"], int(row["GroupSize"]), col))

    # ----------------------------------------------------------------
    # 2. Pack each (stop, date) pool into buses
    # ----------------------------------------------------------------
    # Result: {(stop, date): list of {passengers, members, activity_labels}}
    packed: dict[tuple[str, date], list[dict]] = {}

    for key, members in pool.items():
        buses = _pack_stop(members, bus_min, bus_max)
        packed[key] = buses

    # ----------------------------------------------------------------
    # 3. Multi-stop top-up
    # ----------------------------------------------------------------
    # For each (stop, date) last bus that is under topup_threshold,
    # check adjacent stops' smallest buses for same date.
    processed_topups: set[frozenset] = set()

    for (stop, act_date), buses in packed.items():
        if not buses:
            continue
        last_bus = buses[-1]
        if last_bus["passengers"] >= topup_threshold:
            continue

        for neighbour in adjacency.get(stop, []):
            pair_key = frozenset([(stop, act_date), (neighbour, act_date)])
            if pair_key in processed_topups:
                continue

            neighbour_buses = packed.get((neighbour, act_date), [])
            if not neighbour_buses:
                continue

            # Find the neighbour's smallest bus
            smallest_nb = min(neighbour_buses, key=lambda b: b["passengers"])

            combined = last_bus["passengers"] + smallest_nb["passengers"]
            if combined <= bus_max:
                # Full merge: remove smallest_nb from neighbour, add its members to last_bus
                last_bus["members"].extend(smallest_nb["members"])
                last_bus["passengers"] = combined
                last_bus["activity_labels"] = list(
                    set(last_bus["activity_labels"]) | set(smallest_nb["activity_labels"])
                )
                last_bus["stops"] = list(
                    set(last_bus.get("stops", [stop])) | {neighbour}
                )
                neighbour_buses.remove(smallest_nb)
                processed_topups.add(pair_key)
                assumptions.append(
                    f"TOPUP: Merged {neighbour} bus ({smallest_nb['passengers']} pax) into "
                    f"{stop} bus ({last_bus['passengers'] - smallest_nb['passengers']} pax) "
                    f"on {act_date}. Combined: {last_bus['passengers']} pax."
                )
                break  # only one merge per bus
            else:
                # Partial move: fill last_bus to max, leave remainder in neighbour
                space = bus_max - last_bus["passengers"]
                # Move `space` passengers worth of groups from smallest_nb
                moved = 0
                to_move = []
                for member in sorted(smallest_nb["members"], key=lambda m: m[1]):
                    if moved + member[1] <= space:
                        to_move.append(member)
                        moved += member[1]
                if to_move:
                    for m in to_move:
                        smallest_nb["members"].remove(m)
                        smallest_nb["passengers"] -= m[1]
                    last_bus["members"].extend(to_move)
                    last_bus["passengers"] += moved
                    processed_topups.add(pair_key)

    # ----------------------------------------------------------------
    # 4. Flatten to DataFrame
    # ----------------------------------------------------------------
    trip_rows: list[dict] = []
    trip_id = 1

    for (stop, act_date), buses in sorted(packed.items()):
        for bus_num, bus in enumerate(buses, 1):
            is_under = bus["passengers"] < bus_min
            trip_rows.append({
                "TripID": f"T{trip_id:05d}",
                "Date": act_date,
                "Stop": stop,
                "Stops": "|".join(bus.get("stops", [stop])),
                "ActivityLabels": "|".join(sorted(set(bus["activity_labels"]))),
                "BusNumber": bus_num,
                "Passengers": bus["passengers"],
                "UnderMinimum": is_under,
            })
            trip_id += 1

    bus_trips_df = pd.DataFrame(trip_rows) if trip_rows else pd.DataFrame(
        columns=["TripID", "Date", "Stop", "Stops", "ActivityLabels",
                 "BusNumber", "Passengers", "UnderMinimum"]
    )

    # ----------------------------------------------------------------
    # 5. Stop summary
    # ----------------------------------------------------------------
    stop_summary: dict[str, dict] = {}
    for stop in adjacency:
        subset = bus_trips_df[bus_trips_df["Stop"] == stop] if not bus_trips_df.empty else pd.DataFrame()
        stop_summary[stop] = {
            "total_trips": len(subset),
            "under_min_trips": int(subset["UnderMinimum"].sum()) if not subset.empty else 0,
            "total_passengers": int(subset["Passengers"].sum()) if not subset.empty else 0,
        }

    # ----------------------------------------------------------------
    # 6. Dual-path passenger count validation
    # ----------------------------------------------------------------
    if not bus_trips_df.empty:
        pax_from_buses = int(bus_trips_df["Passengers"].sum())

        # Count from population table: one trip per scheduled activity per group
        pax_from_population = 0
        for _, row in df.iterrows():
            stop = row.get("TransportStop")
            if pd.isna(stop) or stop == "NOT AVAILABLE":
                continue
            for col in act_cols:
                if pd.notna(row[col]):
                    pax_from_population += int(row["GroupSize"])

        if pax_from_buses != pax_from_population:
            assumptions.append(
                f"VALIDATION ERROR: Passenger count mismatch! "
                f"Bus trip total = {pax_from_buses}, "
                f"Population activity sum = {pax_from_population}. "
                "Check for NaN/None confusion in activity assignment columns."
            )
        else:
            assumptions.append(
                f"VALIDATION PASS: Passenger trips match both ways ({pax_from_buses} total)."
            )

    # Summary stats
    if not bus_trips_df.empty:
        total_trips = len(bus_trips_df)
        under_min = int(bus_trips_df["UnderMinimum"].sum())
        pct_under = 100 * under_min / total_trips if total_trips else 0
        assumptions.append(
            f"BUS SUMMARY: {total_trips} total trips, {under_min} under minimum "
            f"({pct_under:.1f}%)."
        )
    else:
        assumptions.append("BUS SUMMARY: No trips generated (no scheduled activities with valid stops).")

    return bus_trips_df, stop_summary, assumptions


# ---------------------------------------------------------------------------
# Bin-packing helper
# ---------------------------------------------------------------------------

def _pack_stop(
    members: list[tuple[str, int, str]],
    bus_min: int,
    bus_max: int,
) -> list[dict]:
    """
    Greedy bin-packing for one stop's members.
    members: list of (GroupID, GroupSize, activity_col)
    Returns list of bus dicts with keys: passengers, members, activity_labels, stops
    """
    total = sum(m[1] for m in members)
    if total == 0:
        return []

    n = max(1, math.ceil(total / bus_max))
    # Reduce bus count where possible (fewer, fuller buses)
    while n > 1 and total / n < bus_min and total / (n - 1) <= bus_max:
        n -= 1

    buses: list[dict] = [
        {"passengers": 0, "members": [], "activity_labels": [], "stops": []}
        for _ in range(n)
    ]

    # Greedy: assign each group (largest first) to least-loaded bus
    for member in sorted(members, key=lambda m: -m[1]):
        group_id, size, act_col = member
        # Find bus with lowest load that can still fit this group
        target = None
        min_load = float("inf")
        for b in buses:
            if b["passengers"] + size <= bus_max and b["passengers"] < min_load:
                min_load = b["passengers"]
                target = b

        if target is None:
            # Open a new bus (overflow)
            target = {"passengers": 0, "members": [], "activity_labels": [], "stops": []}
            buses.append(target)

        target["members"].append(member)
        target["passengers"] += size
        if act_col not in target["activity_labels"]:
            target["activity_labels"].append(act_col)

    return buses
