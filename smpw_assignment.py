"""
SMPW Assignment Module

SMPW is a sub-activity of Field Service. Groups assigned to SMPW have
either their FS1 or FS2 requirement fulfilled (whichever they still need).

Rules:
  - Only groups from SMPW-eligible hotels
  - Only groups whose delegate count is divisible by 2
  - Each group may be assigned SMPW at most once
  - Per-date delegate cap
  - SMPW groups are excluded from bus assignment
  - Bus-efficiency targeting: prefer pulling groups that would cause
    under-minimum bus trips at their stop
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def assign_smpw(
    df: pd.DataFrame,
    smpw_config: dict[str, Any],
    bus_min: int,
    bus_max: int,
    assumptions: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parameters
    ----------
    df : scheduled population with req_* columns already filled by scheduler
    smpw_config : {
        "hotels": [str, ...],          # eligible hotel names
        "dates": [{"date": date, "cap": int}, ...]  # per-date delegate caps
    }
    bus_min, bus_max : bus sizing (used to target under-minimum situations)
    assumptions : running log

    Returns
    -------
    df : updated with columns:
        smpw_date       (date | None)
        smpw_fulfills   ("Field Service 1" | "Field Service 2" | None)
    assumptions : updated log
    """
    df = df.copy()
    df["smpw_date"] = None
    df["smpw_fulfills"] = None

    eligible_hotels = set(smpw_config.get("hotels", []))
    date_configs = sorted(smpw_config.get("dates", []), key=lambda x: x["date"])

    if not eligible_hotels or not date_configs:
        assumptions.append("SMPW: No eligible hotels or dates configured — skipping.")
        return df, assumptions

    fs1_col = "req_Field Service 1"
    fs2_col = "req_Field Service 2"

    if fs1_col not in df.columns and fs2_col not in df.columns:
        assumptions.append("SMPW: No FS1/FS2 columns found — skipping.")
        return df, assumptions

    total_assigned = 0
    total_delegates = 0

    for date_cfg in date_configs:
        smpw_date: date = date_cfg["date"]
        daily_cap: int = date_cfg["cap"]
        delegates_used = 0

        # Candidate groups for this date:
        #   - From eligible hotel
        #   - Even delegate count
        #   - Not already assigned SMPW
        #   - Still need at least one FS (FS1 or FS2 not fulfilled)
        #   - Present on this date (check-in < smpw_date < check-out)
        def needs_fs(row):
            fs1_done = fs1_col in df.columns and pd.notna(row.get(fs1_col))
            fs2_done = fs2_col in df.columns and pd.notna(row.get(fs2_col))
            return not (fs1_done and fs2_done)  # needs at least one

        def which_fs(row):
            fs1_done = fs1_col in df.columns and pd.notna(row.get(fs1_col))
            if not fs1_done:
                return "Field Service 1"
            fs2_done = fs2_col in df.columns and pd.notna(row.get(fs2_col))
            if not fs2_done:
                return "Field Service 2"
            return None

        mask = (
            df["Hotel"].isin(eligible_hotels) &
            (df["GroupSize"] % 2 == 0) &
            df["smpw_date"].isna() &
            (df["CheckIn"] < smpw_date) &
            (df["CheckOut"] > smpw_date)
        )

        # Apply FS need filter
        candidates_idx = [
            idx for idx in df.index[mask]
            if which_fs(df.loc[idx]) is not None
        ]

        if not candidates_idx:
            continue

        candidates = df.loc[candidates_idx].copy()

        # ----------------------------------------------------------------
        # Bus-efficiency targeting:
        # For each stop, calculate how many delegates are going to FS on
        # this date. If the leftover after filling buses is under bus_min,
        # those groups are the primary targets.
        # ----------------------------------------------------------------
        priority_idx: list[int] = []
        secondary_idx: list[int] = []

        # Build stop -> total FS delegates on this date (from non-SMPW groups)
        stop_totals: dict[str, int] = {}
        for _, row in df.iterrows():
            if row.get("smpw_date") is not None:
                continue
            stop = row.get("TransportStop")
            if pd.isna(stop) or stop == "NOT AVAILABLE":
                continue
            # Check if this group has any FS scheduled on this date
            for col in [fs1_col, fs2_col]:
                if col in df.columns and pd.notna(row.get(col)):
                    act_date = row[col]
                    if isinstance(act_date, date) and act_date == smpw_date:
                        stop_totals[stop] = stop_totals.get(stop, 0) + int(row["GroupSize"])

        # For each stop, find the leftover that would cause under-minimum
        stops_with_leftover: dict[str, int] = {}
        for stop, total in stop_totals.items():
            if total == 0:
                continue
            leftover = total % bus_max
            if 0 < leftover < bus_min:
                stops_with_leftover[stop] = leftover

        # Priority: candidates from stops with a leftover problem, smallest size first
        for idx in candidates_idx:
            row = df.loc[idx]
            stop = row.get("TransportStop")
            if stop in stops_with_leftover:
                priority_idx.append(idx)
            else:
                secondary_idx.append(idx)

        # Sort priority candidates: prefer smaller groups (easier to hit exact leftover)
        priority_idx.sort(key=lambda i: df.loc[i, "GroupSize"])
        secondary_idx.sort(key=lambda i: df.loc[i, "GroupSize"])

        ordered_candidates = priority_idx + secondary_idx

        for idx in ordered_candidates:
            if delegates_used >= daily_cap:
                break
            row = df.loc[idx]
            group_size = int(row["GroupSize"])
            if delegates_used + group_size > daily_cap:
                continue  # skip groups that would bust the cap

            fulfills = which_fs(row)
            if fulfills is None:
                continue

            df.at[idx, "smpw_date"] = smpw_date
            df.at[idx, "smpw_fulfills"] = fulfills

            # Mark the FS requirement as fulfilled in the req_ column
            fs_col = "req_Field Service 1" if fulfills == "Field Service 1" else "req_Field Service 2"
            if fs_col in df.columns:
                df.at[idx, fs_col] = smpw_date  # date counts as assignment

            delegates_used += group_size
            total_assigned += 1
            total_delegates += group_size

        assumptions.append(
            f"SMPW [{smpw_date}]: {delegates_used} delegates assigned to SMPW "
            f"(cap={daily_cap})."
        )

    assumptions.append(
        f"SMPW TOTAL: {total_assigned} groups ({total_delegates} delegates) "
        f"diverted to SMPW across {len(date_configs)} dates."
    )

    return df, assumptions


def build_smpw_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame of all SMPW-assigned groups."""
    smpw = df[df["smpw_date"].notna()].copy()
    if smpw.empty:
        return pd.DataFrame(columns=[
            "GroupID", "Hotel", "TransportStop", "GroupSize",
            "SMPW_Date", "Fulfills_FS"
        ])
    return smpw[["GroupID", "Hotel", "TransportStop", "GroupSize",
                 "smpw_date", "smpw_fulfills"]].rename(columns={
        "smpw_date": "SMPW_Date",
        "smpw_fulfills": "Fulfills_FS",
    }).reset_index(drop=True)
