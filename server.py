"""
Convention Logistics Web App — FastAPI server

Run: python3 server.py
Then open: http://localhost:8000

No API key required for the wizard + pipeline.
Optional: set ANTHROPIC_API_KEY for the post-run chat analysis.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import date
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://micahmontana-ui.github.io",
        "https://web-production-b9a7e.up.railway.app",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Per-session chat history (resets on server restart)
chat_history: list[dict] = []
last_result: dict | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api-status")
async def api_status():
    return {"available": bool(API_KEY)}


@app.get("/download/{filename}")
async def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    media = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if filename.endswith(".xlsx") else
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(str(path), media_type=media, filename=filename)


@app.post("/run")
async def run(request: Request):
    """Execute the full pipeline. No API key required."""
    body = await request.json()
    config = body.get("config", {})
    try:
        result = execute_pipeline(config)
        global last_result
        last_result = result
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.post("/chat")
async def chat(request: Request):
    """Stream a Claude response for post-run analysis. Requires ANTHROPIC_API_KEY."""
    if not API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set"}, status_code=403)

    body = await request.json()
    user_msg = body.get("message", "").strip()
    result_context = body.get("result_context")

    if not user_msg:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    chat_history.append({"role": "user", "content": user_msg})

    return StreamingResponse(
        _stream_chat(result_context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
async def reset():
    chat_history.clear()
    global last_result
    last_result = None
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM = """You are a convention logistics analyst embedded in a planning tool called the DSM app. You have full visibility into every step the user has configured.

The context you receive is structured as:
- wizard_config.step1_population: delegate count, group size distribution, length-of-stay distribution, arrival day distribution, window dates
- wizard_config.step2_hotels: hotel names, room capacities, transport stop assignments, bus min/max sizes
- wizard_config.step3_activities: required and optional activities with dates, capacities, priorities, SMPW config, Field Service daily cap
- wizard_config.step4_scheduling: scheduling priority mode, preference depth, same-day pairs, random seed
- pipeline_results: full output including activity assignment rates, daily available delegates (room_block_pct by day), TCO model comparison, bus trips, SMPW counts, non-attendance reasons

Key domain rules to know:
- Convention window is 12 days. Days 7/8/9 are mandatory convention days (not schedulable). Days 6 and 10 are schedulable.
- All delegates must be present for Nights 6, 7, 8, and 9 (check-in by Day 6, check-out no earlier than Day 10).
- Field Service 1 and Field Service 2 share a daily delegate capacity pool.
- SMPW is a sub-activity of Field Service — groups assigned SMPW skip the bus that day.
- EG AM and EG PM are the same Evening Gathering event at different times (mutex group).
- room_block_pct shows what % of total delegates are in town each day (Days 1–12). tco_model_pct is the target occupancy curve to match.

When the user asks about a specific step, reference their exact configured values. When comparing room block to TCO, use the day-by-day numbers. Be concise and specific."""


async def _stream_chat(result_context: dict | None) -> AsyncGenerator[str, None]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)

        system = ANALYSIS_SYSTEM
        if result_context:
            system += f"\n\n---\nFULL APP CONTEXT (wizard configuration + pipeline results):\n{json.dumps(result_context, indent=2)}"

        full_text = ""
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=chat_history,
        ) as stream:
            for event in stream:
                if (hasattr(event, "type") and event.type == "content_block_delta"
                        and hasattr(event, "delta") and event.delta.type == "text_delta"):
                    chunk = event.delta.text
                    full_text += chunk
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

        chat_history.append({"role": "assistant", "content": full_text})
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def execute_pipeline(config: dict) -> dict:
    import numpy as np
    import pandas as pd
    from generate_population import PopulationConfig, generate_population, assign_hotels_stops, HotelConfig
    from build_stop_network import build_stop_network, report_isolated_stop_efficiency
    from activity_definitions import build_activity_defs
    from scheduler import schedule
    from smpw_assignment import assign_smpw, build_smpw_summary
    from bus_assignment import assign_buses
    from build_workbook import build_workbook
    from build_report import build_report

    assumptions: list[str] = []
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)
    assumptions.append(f"RANDOM SEED: {seed}")

    # Population
    pop = config["population"]
    policy = pop.get("checkout_policy", "arrival_based")
    ci_dist = {_parse_date(k): v for k, v in (pop.get("checkin_distribution") or {}).items()}
    co_dist = {_parse_date(k): v for k, v in (pop.get("checkout_distribution") or {}).items()}
    co_fixed = _parse_date(pop["checkout_fixed_date"]) if pop.get("checkout_fixed_date") else None
    from datetime import timedelta
    window_start = _parse_date(pop["window_start"])
    # Convention days (not schedulable): Days 7, 8, 9 from window_start
    # Day 6 and Day 10 are schedulable. Delegates must be present Nights 6–9.
    core_start = window_start + timedelta(days=6)   # Day 7 — first blocked convention day
    core_end   = window_start + timedelta(days=9)   # Day 10 — exclusive upper bound (Day 10 is schedulable)

    pop_cfg = PopulationConfig(
        total_delegates=pop.get("total_delegates"),
        total_groups=pop.get("total_groups"),
        group_size_distribution={int(k): v for k, v in pop["group_size_distribution"].items()},
        nights_distribution={int(k): v for k, v in pop["nights_distribution"].items()},
        core_stay_start=core_start,
        core_stay_end=core_end,
        window_start=window_start,
        window_end=_parse_date(pop["window_end"]),
        checkout_policy=policy,
        checkin_distribution=ci_dist,
        checkout_fixed_date=co_fixed,
        checkout_distribution=co_dist,
        random_seed=seed,
    )

    df, assumptions = generate_population(pop_cfg, rng)

    hotel_cfg = HotelConfig(hotels=config["hotels"], hotel_to_stop=config["hotel_to_stop"])
    df, assumptions = assign_hotels_stops(df, hotel_cfg, rng, assumptions)

    adjacency, assumptions = build_stop_network(
        config["stop_names"],
        [tuple(p) for p in config.get("pairing_rules", [])],
        assumptions,
    )

    # Activities
    # Field Service shared delegate pool — both FS1 and FS2 draw from this per-day cap
    fs_cap_raw = config.get("field_service_cap", {})
    fs_cap = {_parse_date(k): int(v) for k, v in fs_cap_raw.items() if v}
    cap_pools = {"Field Service": fs_cap} if fs_cap else None

    req_raw = []
    for act in config.get("required_activities", []):
        cap = act.get("capacity")
        if isinstance(cap, dict):
            cap = {_parse_date(k): v for k, v in cap.items()}
        is_fs = act["name"] in ("Field Service 1", "Field Service 2")
        req_raw.append({
            "name": act["name"],
            "dates": [_parse_date(d) for d in act.get("dates", [])],
            "capacity": None if is_fs and fs_cap else cap,  # pool replaces individual cap
            "priority": act.get("priority", 999),
            "mutex_group": act.get("mutex_group"),
            "prerequisites": act.get("prerequisites") or [],
            "cap_pool": "Field Service" if is_fs and fs_cap else None,
        })

    opt_raw = []
    for act in config.get("optional_activities", []):
        cap = act.get("capacity")
        if isinstance(cap, dict):
            cap = {_parse_date(k): v for k, v in cap.items()}
        opt_raw.append({
            "name": act["name"],
            "dates": [_parse_date(d) for d in act.get("dates", [])],
            "capacity": cap,
            "min_capacity": act.get("min_capacity"),
            "want_pct": act.get("want_pct", 0),
            "prerequisite": act.get("prerequisite"),
            "prerequisite_group": act.get("prerequisite_group"),
        })

    REQUIRED, OPTIONAL, assumptions = build_activity_defs(
        req_raw, opt_raw, pop_cfg.window_start, pop_cfg.window_end, assumptions
    )

    same_day_pairs = [
        frozenset(p) for p in config.get("same_day_pairs", [])
        if len(p) == 2
    ]

    pd_raw = config.get("preference_depth", {})
    preference_depth = {
        "three_plus": int(pd_raw.get("threePlus", 40)),
        "two_three":  int(pd_raw.get("twoThree",  30)),
        "one_two":    int(pd_raw.get("oneTwo",    20)),
        "one":        int(pd_raw.get("one",       10)),
    }

    df, non_attendance, assumptions = schedule(
        df, REQUIRED, OPTIONAL,
        pop_cfg.core_stay_start, pop_cfg.core_stay_end,
        rng, assumptions,
        order_is_priority=config.get("order_is_priority", True),
        same_day_pairs=same_day_pairs,
        preference_depth=preference_depth,
        cap_pools=cap_pools,
    )

    bus_min = config["bus_min"]
    bus_max = config["bus_max"]
    topup = config.get("topup_threshold", bus_max)

    # SMPW assignment (post-schedule, pre-bus)
    smpw_cfg_raw = config.get("smpw", {})
    smpw_config = {
        "hotels": smpw_cfg_raw.get("hotels", []),
        "dates": [
            {"date": _parse_date(d["date"]), "cap": int(d["cap"])}
            for d in smpw_cfg_raw.get("dates", [])
            if d.get("date") and d.get("cap")
        ],
    }
    df, assumptions = assign_smpw(df, smpw_config, bus_min, bus_max, assumptions)
    smpw_df = build_smpw_summary(df)

    # Bus assignment — exclude SMPW groups (they don't ride buses)
    df_for_buses = df[df["smpw_date"].isna()].copy()
    bus_trips_df, stop_summary, assumptions = assign_buses(
        df_for_buses, adjacency, bus_min, bus_max, topup, assumptions
    )
    report_isolated_stop_efficiency(adjacency, stop_summary, assumptions)

    # Outputs
    wb_path = str(OUTPUT_DIR / "convention_logistics.xlsx")
    build_workbook(
        wb_path, df, bus_trips_df, stop_summary, non_attendance,
        REQUIRED, OPTIONAL, config["hotels"], assumptions, smpw_df=smpw_df,
    )

    total_trips = len(bus_trips_df) if not bus_trips_df.empty else 0
    under_min = int(bus_trips_df["UnderMinimum"].sum()) if not bus_trips_df.empty else 0
    pct_under = f"{100*under_min/total_trips:.1f}" if total_trips else "0.0"

    # Activity assignment — merge mutex groups (e.g. EG AM + EG PM → single "EG" entry)
    raw_req = {
        name: {
            "groups": int(df[f"req_{name}"].notna().sum()),
            "delegates": int(df.loc[df[f"req_{name}"].notna(), "GroupSize"].sum()),
        } for name in REQUIRED
    }
    mutex_groups: dict[str, list[str]] = {}
    for name, act in REQUIRED.items():
        mg = act.get("mutex_group")
        if mg:
            mutex_groups.setdefault(mg, []).append(name)

    act_assigned: dict = {}
    merged_names: set = set()
    for mg, names in mutex_groups.items():
        act_assigned[f"[R] {mg}"] = {
            "groups": sum(raw_req[n]["groups"] for n in names),
            "delegates": sum(raw_req[n]["delegates"] for n in names),
            "breakdown": {n: raw_req[n] for n in names},
        }
        merged_names.update(names)
    for name in REQUIRED:
        if name not in merged_names:
            act_assigned[f"[R] {name}"] = raw_req[name]
    for name in OPTIONAL:
        act_assigned[f"[O] {name}"] = {
            "groups": int(df[f"opt_{name}"].notna().sum()),
            "delegates": int(df.loc[df[f"opt_{name}"].notna(), "GroupSize"].sum()),
        }

    # SMPW totals and breakdown by FS activity
    smpw_total_groups = int(df["smpw_date"].notna().sum())
    smpw_total_delegates = int(df.loc[df["smpw_date"].notna(), "GroupSize"].sum()) if smpw_total_groups else 0
    smpw_by_day: dict = {}
    smpw_by_fs: dict = {}
    if smpw_total_groups:
        for d, grp in df[df["smpw_date"].notna()].groupby("smpw_date"):
            fulfills_vals = grp["smpw_fulfills"].dropna().unique().tolist()
            smpw_by_day[str(d)] = {
                "groups": int(len(grp)),
                "delegates": int(grp["GroupSize"].sum()),
                "fulfills": fulfills_vals[0] if len(fulfills_vals) == 1 else ", ".join(str(v) for v in fulfills_vals),
            }
        for fs_name, grp in df[df["smpw_date"].notna()].groupby("smpw_fulfills"):
            if fs_name:
                smpw_by_fs[str(fs_name)] = {"groups": int(len(grp)), "delegates": int(grp["GroupSize"].sum())}

    # Multi-stop bus trips
    multi_stop_trips = 0
    if not bus_trips_df.empty and "Stops" in bus_trips_df.columns:
        multi_stop_trips = int((bus_trips_df["Stops"].str.contains("|", regex=False)).sum())

    # One-way bus trips broken down by FS1 and FS2
    fs_bus_trips: dict = {}
    cong_service_days = 0
    if not bus_trips_df.empty and "ActivityLabels" in bus_trips_df.columns:
        for fs_name in ["Field Service 1", "Field Service 2"]:
            n = int(bus_trips_df["ActivityLabels"].str.contains(fs_name, na=False).sum())
            fs_bus_trips[fs_name] = n
            cong_service_days += n

    # Delegates with two activities on the same day
    import pandas as _pd
    act_cols = [c for c in df.columns if c.startswith("req_") or c.startswith("opt_")]
    def _has_same_day(row):
        dates = [str(row[c]) for c in act_cols if _pd.notna(row[c])]
        return len(dates) != len(set(dates))
    same_day_mask = df.apply(_has_same_day, axis=1)
    double_booked_groups = int(same_day_mask.sum())
    double_booked_delegates = int(df.loc[same_day_mask, "GroupSize"].sum())

    # Daily breakdown: date → {activity_name → {groups, delegates}}
    from collections import defaultdict
    daily_activity: dict = defaultdict(dict)
    for name in REQUIRED:
        col = f"req_{name}"
        for date_val, grp in df[df[col].notna()].groupby(col):
            daily_activity[str(date_val)][name] = {
                "groups": int(len(grp)),
                "delegates": int(grp["GroupSize"].sum()),
            }
    for name in OPTIONAL:
        col = f"opt_{name}"
        for date_val, grp in df[df[col].notna()].groupby(col):
            daily_activity[str(date_val)][name] = {
                "groups": int(len(grp)),
                "delegates": int(grp["GroupSize"].sum()),
            }
    for date_val, grp in df[df["smpw_date"].notna()].groupby("smpw_date"):
        daily_activity[str(date_val)]["SMPW"] = {
            "groups": int(len(grp)),
            "delegates": int(grp["GroupSize"].sum()),
        }

    # Per-day available (ci < day < co) for every day in the full window
    from datetime import timedelta as _td
    all_act_date_cols = [f"req_{n}" for n in REQUIRED] + [f"opt_{n}" for n in OPTIONAL]
    daily_available: dict = {}
    daily_scheduled_delegates: dict = {}

    # Compute available for every day in window (not just activity days)
    _d = pop_cfg.window_start
    while _d <= pop_cfg.window_end:
        avail_mask = (df["CheckIn"] <= _d) & (df["CheckOut"] > _d)
        daily_available[str(_d)] = int(df.loc[avail_mask, "GroupSize"].sum())
        _d += _td(days=1)

    for d_str in daily_activity:
        d = date.fromisoformat(d_str)
        avail_mask = (df["CheckIn"] <= d) & (df["CheckOut"] > d)
        daily_available[d_str] = int(df.loc[avail_mask, "GroupSize"].sum())
        sched_mask = _pd.Series(False, index=df.index)
        for col in all_act_date_cols:
            if col in df.columns:
                sched_mask = sched_mask | (df[col] == d)
        if "smpw_date" in df.columns:
            sched_mask = sched_mask | (df["smpw_date"] == d)
        daily_scheduled_delegates[d_str] = int(df.loc[sched_mask, "GroupSize"].sum())

    summary = {
        "total_groups": len(df),
        "total_delegates": int(df["GroupSize"].sum()),
        "total_trips": total_trips,
        "under_min_trips": under_min,
        "pct_under_min": pct_under,
        "activity_assigned": act_assigned,
        "smpw_groups": smpw_total_groups,
        "smpw_delegates": smpw_total_delegates,
        "smpw_by_day": smpw_by_day,
        "smpw_by_fs": smpw_by_fs,
        "multi_stop_trips": multi_stop_trips,
        "cong_service_days": cong_service_days,
        "fs_bus_trips": fs_bus_trips,
        "double_booked_groups": double_booked_groups,
        "double_booked_delegates": double_booked_delegates,
        "daily_activity": dict(daily_activity),
        "daily_available": daily_available,
        "daily_scheduled_delegates": daily_scheduled_delegates,
        "activity_names": list(REQUIRED.keys()) + list(OPTIONAL.keys()) + (["SMPW"] if smpw_total_groups else []),
        "window_start": str(pop_cfg.window_start),
        "window_end": str(pop_cfg.window_end),
        "biggest_findings": [a for a in assumptions if a.startswith(("WARNING", "ISOLATED", "VALIDATION ERROR"))],
    }

    report_path = str(OUTPUT_DIR / "convention_logistics_report.docx")
    build_report(report_path, summary, assumptions, non_attendance, stop_summary, bus_trips_df, REQUIRED)

    return {
        "summary": summary,
        "stop_summary": stop_summary,
        "excel_url": "/download/convention_logistics.xlsx",
        "report_url": "/download/convention_logistics_report.docx",
    }


# ---------------------------------------------------------------------------
# Static files + entry point
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n  Convention Logistics Modeler")
    print("  → http://localhost:8000\n")
    if not API_KEY:
        print("  ℹ  No ANTHROPIC_API_KEY set — wizard and pipeline work fine,")
        print("     but post-run chat analysis will be disabled.\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
