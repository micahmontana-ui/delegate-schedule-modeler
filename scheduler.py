"""
Module 5: Scheduling Engine

Supports:
  - prerequisites for required activities (multi-prerequisite list)
  - mutex_group: activities in same group — only one assigned per group
  - prerequisite_group: optional activities satisfied by any assignment in a mutex group
  - same_day_pairs: frozensets of activity names allowed to share a day
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

SCHEDULE_ORDER_IS_PRIORITY_NOT_CALENDAR = True


def schedule(
    df: pd.DataFrame,
    required_activities: dict[str, dict],
    optional_activities: dict[str, dict],
    core_stay_start: date,
    core_stay_end: date,
    rng: np.random.Generator,
    assumptions: list[str],
    order_is_priority: bool = SCHEDULE_ORDER_IS_PRIORITY_NOT_CALENDAR,
    same_day_pairs: list[frozenset] | None = None,
    preference_depth: dict | None = None,
    cap_pools: dict[str, dict[date, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = df.copy()
    same_day_pairs = same_day_pairs or []
    preference_depth = preference_depth or {"three_plus": 40, "two_three": 30, "one_two": 20, "one": 10}
    pool_counters: dict[str, dict[date, int]] = {name: dict(dates) for name, dates in (cap_pools or {}).items()}

    # ── Pre-assign want counts per group ──────────────────────────────────────
    n_opt = len(optional_activities)
    pd_total = sum(preference_depth.values()) or 100
    pd_weights = [
        preference_depth.get("three_plus", 40) / pd_total,
        preference_depth.get("two_three",  30) / pd_total,
        preference_depth.get("one_two",    20) / pd_total,
        preference_depth.get("one",        10) / pd_total,
    ]
    pd_choices = rng.choice([0, 1, 2, 3], size=len(df), p=pd_weights)
    group_want_counts: list[int] = []
    for tier in pd_choices:
        if tier == 0:   # 3+
            group_want_counts.append(int(rng.integers(3, max(4, n_opt + 1))))
        elif tier == 1: # 2–3
            group_want_counts.append(int(rng.integers(2, 4)))
        elif tier == 2: # 1–2
            group_want_counts.append(int(rng.integers(1, 3)))
        else:           # exactly 1
            group_want_counts.append(1)

    assumptions.append(
        f"PREFERENCE DEPTH: tiers assigned to {len(df)} groups — "
        f"3+: {int(pd_weights[0]*100)}%, 2-3: {int(pd_weights[1]*100)}%, "
        f"1-2: {int(pd_weights[2]*100)}%, 1: {int(pd_weights[3]*100)}%."
    )

    # Build same-day lookup: act_name -> set of act_names it can share a day with
    same_day_map: dict[str, set[str]] = {}
    for pair in same_day_pairs:
        pair_list = list(pair)
        if len(pair_list) == 2:
            same_day_map.setdefault(pair_list[0], set()).add(pair_list[1])
            same_day_map.setdefault(pair_list[1], set()).add(pair_list[0])

    req_sorted = sorted(required_activities.items(), key=lambda x: x[1]["priority"])

    assumptions.append(
        f"SCHEDULING: order_is_priority={order_is_priority}. "
        + ("Allocation priority — capacity-limited activities get first pick."
           if order_is_priority else "Calendar order enforced.")
    )
    assumptions.append(
        "SCHEDULING: Required activity order: "
        + ", ".join(f"{n}(p={v['priority']})" for n, v in req_sorted)
    )
    if same_day_pairs:
        assumptions.append(
            "SAME-DAY EXCEPTIONS: " + "; ".join(
                " + ".join(sorted(p)) for p in same_day_pairs
            )
        )

    # Add assignment columns
    for act_name in required_activities:
        df[f"req_{act_name}"] = None
    for act_name in optional_activities:
        df[f"opt_{act_name}"] = None

    # Capacity counters
    cap_counters: dict[str, dict[date, int | None]] = {}
    for act_name, act in {**required_activities, **optional_activities}.items():
        cap = act["capacity"]
        if cap is None:
            cap_counters[act_name] = {d: None for d in act["dates"]}
        elif isinstance(cap, dict):
            cap_counters[act_name] = dict(cap)
        else:
            cap_counters[act_name] = {d: cap for d in act["dates"]}

    non_attendance_rows: list[dict] = []

    core_days: set[date] = set()
    d = core_stay_start
    while d < core_stay_end:  # core_stay_end is the checkout morning, not a blocked night
        core_days.add(d)
        d += timedelta(days=1)

    for idx in rng.permutation(len(df)).tolist():
        row = df.iloc[idx]
        group_id = row["GroupID"]
        ci: date = row["CheckIn"]
        co: date = row["CheckOut"]

        candidate_days: set[date] = set()
        d = ci + timedelta(days=1)
        while d < co:  # exclude arrival day; exclude checkout day (must be in town: CheckOut > date)
            if d not in core_days:
                candidate_days.add(d)
            d += timedelta(days=1)

        # Track: date -> which activity is assigned on that day (for same-day checking)
        day_to_activity: dict[date, str] = {}
        # Track: mutex_group -> activity name assigned for this group
        mutex_assigned: dict[str, str] = {}

        # ----------------------------------------------------------------
        # Step B: Required activities
        # ----------------------------------------------------------------
        for act_name, act_def in req_sorted:

            # Mutex check — skip if another activity in this group is already assigned
            mutex = act_def.get("mutex_group")
            if mutex and mutex in mutex_assigned:
                continue

            # Prerequisite check — all named required activities must be assigned
            prereqs = act_def.get("prerequisites") or []
            unmet = [p for p in prereqs if pd.isna(df.at[idx, f"req_{p}"])]
            if unmet:
                non_attendance_rows.append({
                    "GroupID": group_id,
                    "Activity": act_name,
                    "Reason": f"prerequisite not met: {unmet}",
                    "EligibleDays": len(candidate_days),
                    "AssignedActivities": _count_assigned(df, idx),
                })
                continue

            # Available dates: in candidate window, not already taken
            # (unless a same-day exception applies between this activity and the one on that day)
            allowed_same_day = same_day_map.get(act_name, set())
            available_dates = [
                d for d in act_def["dates"]
                if d in candidate_days and (
                    d not in day_to_activity or
                    day_to_activity[d] in allowed_same_day
                )
            ]

            if not available_dates:
                non_attendance_rows.append({
                    "GroupID": group_id,
                    "Activity": act_name,
                    "Reason": f"structural: 0 eligible days (check-in {ci}, check-out {co}, "
                              f"core-stay {core_stay_start}–{core_stay_end})",
                    "EligibleDays": len(candidate_days),
                    "AssignedActivities": _count_assigned(df, idx),
                })
                continue

            cap = cap_counters[act_name]
            group_size = int(row["GroupSize"])
            cap_pool_name = act_def.get("cap_pool")
            pool_cap = pool_counters.get(cap_pool_name) if cap_pool_name else None
            chosen = _pick_date(available_dates, cap, order_is_priority, group_size, pool_cap)

            if chosen is None:
                non_attendance_rows.append({
                    "GroupID": group_id,
                    "Activity": act_name,
                    "Reason": "capacity exhausted on all eligible dates",
                    "EligibleDays": len(available_dates),
                    "AssignedActivities": _count_assigned(df, idx),
                })
                continue

            df.at[idx, f"req_{act_name}"] = chosen
            day_to_activity[chosen] = act_name
            if cap.get(chosen) is not None:
                cap_counters[act_name][chosen] -= group_size
            if pool_cap is not None and pool_cap.get(chosen) is not None:
                pool_counters[cap_pool_name][chosen] -= group_size
            if mutex:
                mutex_assigned[mutex] = act_name

        # ----------------------------------------------------------------
        # Step C: Optional activities — preference-ranked assignment
        # ----------------------------------------------------------------
        leftover_days = candidate_days - set(day_to_activity.keys())
        if not leftover_days or n_opt == 0:
            continue

        want_count = group_want_counts[idx]
        if want_count == 0:
            continue

        # Build this group's ranked preference list:
        # score = popularity * noise, then sort descending so most-wanted is first
        pref_list = sorted(
            optional_activities.items(),
            key=lambda kv: kv[1]["want_pct"] * (0.5 + rng.random()),
            reverse=True,
        )

        scheduled_opt = 0
        for act_name, act_def in pref_list:
            if scheduled_opt >= want_count:
                break

            # Prerequisite: single required activity
            prereq = act_def.get("prerequisite")
            if prereq:
                if pd.isna(df.at[idx, f"req_{prereq}"]):
                    continue

            # Prerequisite: mutex group (EG)
            prereq_group = act_def.get("prerequisite_group")
            if prereq_group and prereq_group not in mutex_assigned:
                continue

            allowed_same_day = same_day_map.get(act_name, set())
            available = [
                d for d in act_def["dates"]
                if d in candidate_days and (
                    d not in day_to_activity or
                    day_to_activity[d] in allowed_same_day
                )
            ]
            if not available:
                continue

            cap = cap_counters[act_name]
            chosen = _pick_date(available, cap, order_is_priority=True, group_size=int(row["GroupSize"]))
            if chosen is None:
                continue

            df.at[idx, f"opt_{act_name}"] = chosen
            day_to_activity[chosen] = act_name
            if cap.get(chosen) is not None:
                cap_counters[act_name][chosen] -= int(row["GroupSize"])
            scheduled_opt += 1

    # ----------------------------------------------------------------
    # Step D: Optional activity minimum-capacity cancellation
    # If total delegates assigned to an optional activity falls below its
    # min_capacity, cancel the activity entirely and free those days.
    # ----------------------------------------------------------------
    for act_name, act_def in optional_activities.items():
        min_cap = act_def.get("min_capacity")
        if not min_cap:
            continue
        col = f"opt_{act_name}"
        assigned_delegates = int(df.loc[df[col].notna(), "GroupSize"].sum())
        if assigned_delegates < min_cap:
            df[col] = None
            assumptions.append(
                f"OPTIONAL CANCELLED [{act_name}]: only {assigned_delegates} delegates "
                f"assigned, below minimum of {min_cap}. Activity cancelled."
            )

    # ----------------------------------------------------------------
    # Step E: Non-attendance summary
    # ----------------------------------------------------------------
    non_attendance = pd.DataFrame(non_attendance_rows)
    if non_attendance.empty:
        non_attendance = pd.DataFrame(
            columns=["GroupID", "Activity", "Reason", "EligibleDays", "AssignedActivities"]
        )

    for act_name in required_activities:
        mutex = required_activities[act_name].get("mutex_group")
        n_assigned = df[f"req_{act_name}"].notna().sum()
        n_not = len(df) - n_assigned
        structural = 0
        if not non_attendance.empty:
            structural = len(non_attendance[
                (non_attendance["Activity"] == act_name) &
                (non_attendance["Reason"].str.startswith("structural"))
            ])
        mutex_note = f" [mutex: {mutex}]" if mutex else ""
        assumptions.append(
            f"SCHEDULE RESULT [{act_name}]{mutex_note}: {n_assigned} assigned, "
            f"{n_not} not scheduled ({structural} structural, {n_not-structural} other)."
        )

    return df, non_attendance, assumptions


def _count_assigned(df: pd.DataFrame, idx: int) -> int:
    return sum(1 for col in df.columns if col.startswith("req_") and pd.notna(df.at[idx, col]))


def _pick_date(
    available_dates: list[date],
    cap: dict[date, int | None],
    order_is_priority: bool,
    group_size: int = 1,
    pool_cap: dict[date, int] | None = None,
) -> date | None:
    def _effective_remaining(d: date) -> float:
        own = cap.get(d)
        pool = pool_cap.get(d) if pool_cap else None
        if own is None and pool is None:
            return float("inf")
        if own is None:
            return float(pool)
        if pool is None:
            return float(own)
        return float(min(own, pool))

    def _fits(d: date) -> bool:
        return _effective_remaining(d) >= group_size

    if order_is_priority:
        chosen = max(available_dates, key=_effective_remaining)
        return chosen if _fits(chosen) else None
    else:
        for d in sorted(available_dates):
            if _fits(d):
                return d
        return None
