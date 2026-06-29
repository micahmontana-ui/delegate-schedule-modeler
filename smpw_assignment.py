"""
SMPW Assignment Module

SMPW is a sub-activity of Field Service. Groups assigned to SMPW have
either their FS1 or FS2 requirement fulfilled (whichever is scheduled on
the SMPW date). They do not ride a bus that day.

Rules:
  - Only groups from SMPW-eligible hotels
  - Only groups present on the SMPW date (CheckIn < date < CheckOut)
  - Group must already have FS1 or FS2 scheduled on that exact date
    (SMPW replaces the bus trip — the FS col date is unchanged)
  - Each group may be assigned SMPW at most once
  - Per-date delegate cap (greedy smallest-first to fill cap tightly)
  - Bus-efficiency targeting: prefer stops with under-minimum leftovers
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

        # --- Eligibility -------------------------------------------------
        # Group must:
        #   1. Be from an eligible hotel
        #   2. Be present (CheckIn < smpw_date < CheckOut)
        #   3. Not already assigned SMPW
        #   4. Have FS1 or FS2 scheduled on this exact date
        #      (SMPW replaces that bus trip — the FS col is left unchanged)

        def which_fs_on_date(row) -> str | None:
            if fs1_col in df.columns and row.get(fs1_col) == smpw_date:
                return "Field Service 1"
            if fs2_col in df.columns and row.get(fs2_col) == smpw_date:
                return "Field Service 2"
            return None

        base_mask = (
            df["Hotel"].isin(eligible_hotels) &
            df["smpw_date"].isna() &
            (df["CheckIn"] < smpw_date) &
            (df["CheckOut"] > smpw_date)
        )

        candidates_idx = [
            idx for idx in df.index[base_mask]
            if which_fs_on_date(df.loc[idx]) is not None
        ]

        if not candidates_idx:
            assumptions.append(
                f"SMPW [{smpw_date}]: 0 delegates assigned (cap={daily_cap}) "
                f"— no eligible groups have FS scheduled on this date."
            )
            continue

        candidates = df.loc[candidates_idx].copy()

        # --- Bus-efficiency targeting ------------------------------------
        # Build stop → total FS delegates scheduled on this date
        stop_totals: dict[str, int] = {}
        for idx2, row2 in df.iterrows():
            if row2.get("smpw_date") is not None:
                continue
            stop = row2.get("TransportStop")
            if pd.isna(stop) or stop == "NOT AVAILABLE":
                continue
            for col in [fs1_col, fs2_col]:
                if col in df.columns and row2.get(col) == smpw_date:
                    stop_totals[stop] = stop_totals.get(stop, 0) + int(row2["GroupSize"])

        stops_with_leftover: set[str] = set()
        for stop, total in stop_totals.items():
            leftover = total % bus_max
            if 0 < leftover < bus_min:
                stops_with_leftover.add(stop)

        priority_idx: list[int] = []
        secondary_idx: list[int] = []
        for idx in candidates_idx:
            stop = df.loc[idx, "TransportStop"]
            if stop in stops_with_leftover:
                priority_idx.append(idx)
            else:
                secondary_idx.append(idx)

        # Sort each tier smallest-first for tight greedy cap-filling
        priority_idx.sort(key=lambda i: df.loc[i, "GroupSize"])
        secondary_idx.sort(key=lambda i: df.loc[i, "GroupSize"])
        ordered_candidates = priority_idx + secondary_idx

        # --- Greedy fill -------------------------------------------------
        # Try every candidate; skip only if it would bust the cap.
        # Do NOT stop early — smaller groups later may still fit.
        for idx in ordered_candidates:
            if delegates_used >= daily_cap:
                break
            group_size = int(df.loc[idx, "GroupSize"])
            if delegates_used + group_size > daily_cap:
                continue  # too big right now; keep trying smaller ones

            fulfills = which_fs_on_date(df.loc[idx])
            if fulfills is None:
                continue

            df.at[idx, "smpw_date"] = smpw_date
            df.at[idx, "smpw_fulfills"] = fulfills
            # FS col already has smpw_date — no change needed
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
