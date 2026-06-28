"""
Module 4: Activity Definitions

Pure function interface:
    build_activity_defs(raw_required, raw_optional, window_start, window_end, assumptions)
        -> (REQUIRED_ACTIVITIES, OPTIONAL_ACTIVITIES, assumptions)
"""

from __future__ import annotations

from datetime import date
from typing import Any


def build_activity_defs(
    raw_required: list[dict[str, Any]],
    raw_optional: list[dict[str, Any]],
    window_start: date,
    window_end: date,
    assumptions: list[str],
) -> tuple[dict, dict, list[str]]:
    """
    Parameters
    ----------
    raw_required : list of dicts with keys:
        name, dates (list[date]), capacity (dict[date,int] | None), priority (int)
    raw_optional : list of dicts with keys:
        name, dates (list[date]), capacity (dict[date,int] | None),
        want_pct (float 0-1), prerequisite (str | None)

    Returns
    -------
    REQUIRED_ACTIVITIES : {name: {dates, capacity, priority}}
    OPTIONAL_ACTIVITIES : {name: {dates, capacity, want_pct, prerequisite}}
    """
    REQUIRED: dict[str, dict] = {}
    OPTIONAL: dict[str, dict] = {}

    all_dates_in_window = _date_range(window_start, window_end)

    for act in raw_required:
        name = act["name"]
        dates = act.get("dates", [])
        capacity = act.get("capacity")  # None means unlimited; keep distinct from 0
        priority = act.get("priority", 999)

        # Validate dates
        out_of_window = [d for d in dates if d not in all_dates_in_window]
        if out_of_window:
            assumptions.append(
                f"DATE WARNING [{name}]: {len(out_of_window)} date(s) outside stay window "
                f"({window_start}–{window_end}): {out_of_window}. These dates will be unusable."
            )

        if capacity is None:
            assumptions.append(
                f"CAPACITY [{name}]: No capacity stated — treated as UNLIMITED. "
                "If a cap exists, provide it to enable load-balancing."
            )
        elif isinstance(capacity, dict):
            missing_dates = [d for d in dates if d not in capacity]
            if missing_dates:
                assumptions.append(
                    f"CAPACITY [{name}]: No per-date capacity given for {missing_dates}. "
                    "These dates treated as unlimited."
                )

        REQUIRED[name] = {
            "dates": dates,
            "capacity": capacity,
            "priority": priority,
            "prerequisites": act.get("prerequisites") or [],
            "mutex_group": act.get("mutex_group"),
        }

    for act in raw_optional:
        name = act["name"]
        dates = act.get("dates", [])
        capacity = act.get("capacity")
        want_pct = act.get("want_pct", 0.0)
        prereq = act.get("prerequisite")

        out_of_window = [d for d in dates if d not in all_dates_in_window]
        if out_of_window:
            assumptions.append(
                f"DATE WARNING [{name}]: {len(out_of_window)} date(s) outside stay window: "
                f"{out_of_window}."
            )

        if capacity is None:
            assumptions.append(
                f"CAPACITY [{name}] (optional): No capacity stated — treated as UNLIMITED."
            )

        if prereq:
            if prereq not in {a["name"] for a in raw_required} | {a["name"] for a in raw_optional}:
                assumptions.append(
                    f"PREREQUISITE WARNING [{name}]: prerequisite '{prereq}' not found in any "
                    "activity definition. Prerequisite check will always fail — groups won't be "
                    "eligible for this activity unless you fix the prerequisite name."
                )

        OPTIONAL[name] = {
            "dates": dates,
            "capacity": capacity,
            "want_pct": want_pct,
            "prerequisite": prereq,
            "prerequisite_group": act.get("prerequisite_group"),
        }

    assumptions.append(
        f"ACTIVITIES: {len(REQUIRED)} required, {len(OPTIONAL)} optional defined."
    )
    return REQUIRED, OPTIONAL, assumptions


def _date_range(start: date, end: date) -> set[date]:
    days = (end - start).days + 1
    return {start + __import__("datetime").timedelta(days=i) for i in range(days)}
