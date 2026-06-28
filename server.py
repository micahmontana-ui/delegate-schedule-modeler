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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

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
    return FileResponse(BASE / "static" / "index.html")


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

ANALYSIS_SYSTEM = """You are a convention logistics analyst. The user has just run a logistics modeling pipeline and wants help interpreting the results.

Help them understand:
- Activity assignment rates and non-attendance reasons (structural vs capacity-limited)
- Bus efficiency — what's driving under-minimum trips, which stops are isolated
- Whether the scheduling priority mode (allocation vs calendar) produced the expected results
- Trade-offs if they want to re-run with different parameters
- What the Excel sheets and Word report contain

Be concise and specific. Reference actual numbers from the results when answering."""


async def _stream_chat(result_context: dict | None) -> AsyncGenerator[str, None]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)

        system = ANALYSIS_SYSTEM
        if result_context:
            system += f"\n\nPipeline results summary:\n{json.dumps(result_context, indent=2)}"

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
    # Nights 7, 8, 9 are mandatory convention nights (0-indexed: days 6, 7, 8 from window_start)
    core_start = window_start + timedelta(days=6)   # start of night 7
    core_end   = window_start + timedelta(days=9)   # morning after night 9 (groups must not check out before this)

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
    req_raw = []
    for act in config.get("required_activities", []):
        cap = act.get("capacity")
        if isinstance(cap, dict):
            cap = {_parse_date(k): v for k, v in cap.items()}
        req_raw.append({
            "name": act["name"],
            "dates": [_parse_date(d) for d in act.get("dates", [])],
            "capacity": cap,
            "priority": act.get("priority", 999),
            "mutex_group": act.get("mutex_group"),
            "prerequisites": act.get("prerequisites") or [],
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
    act_assigned = {name: int(df[f"req_{name}"].notna().sum()) for name in REQUIRED}
    smpw_total_groups = int(df["smpw_date"].notna().sum())
    smpw_total_delegates = int(df.loc[df["smpw_date"].notna(), "GroupSize"].sum()) if smpw_total_groups else 0

    # Multi-stop bus trips
    multi_stop_trips = 0
    if not bus_trips_df.empty and "Stops" in bus_trips_df.columns:
        multi_stop_trips = int((bus_trips_df["Stops"].str.contains("|", regex=False)).sum())

    # Congregation service days = total FS1 + FS2 bus trips
    cong_service_days = 0
    if not bus_trips_df.empty and "ActivityLabels" in bus_trips_df.columns:
        fs_mask = bus_trips_df["ActivityLabels"].str.contains("Field Service", na=False)
        cong_service_days = int(fs_mask.sum())

    # Delegates with two activities on the same day
    import pandas as _pd
    act_cols = [c for c in df.columns if c.startswith("req_") or c.startswith("opt_")]
    def _has_same_day(row):
        dates = [str(row[c]) for c in act_cols if _pd.notna(row[c])]
        return len(dates) != len(set(dates))
    same_day_mask = df.apply(_has_same_day, axis=1)
    double_booked_groups = int(same_day_mask.sum())
    double_booked_delegates = int(df.loc[same_day_mask, "GroupSize"].sum())

    summary = {
        "total_groups": len(df),
        "total_delegates": int(df["GroupSize"].sum()),
        "total_trips": total_trips,
        "under_min_trips": under_min,
        "pct_under_min": pct_under,
        "activity_assigned": act_assigned,
        "smpw_groups": smpw_total_groups,
        "smpw_delegates": smpw_total_delegates,
        "multi_stop_trips": multi_stop_trips,
        "cong_service_days": cong_service_days,
        "double_booked_groups": double_booked_groups,
        "double_booked_delegates": double_booked_delegates,
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
