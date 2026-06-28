"""
Module 3: Bus Stop Network

Pure function interface:
    build_stop_network(stop_defs, pairing_rules, assumptions) -> (adjacency_dict, assumptions)
    report_isolated_stops(adjacency_dict, bus_results) -> list[str]
"""

from __future__ import annotations

from typing import Any


def build_stop_network(
    stop_names: list[str],
    pairing_rules: list[tuple[str, str]],
    assumptions: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Parameters
    ----------
    stop_names : list of all stop names in the system
    pairing_rules : list of (stop_a, stop_b) pairs the user explicitly stated
    assumptions : running assumptions log (mutated in place)

    Returns
    -------
    adjacency : bidirectional adjacency dict
    assumptions : updated log
    """
    # Initialise with empty lists
    adjacency: dict[str, list[str]] = {s: [] for s in stop_names}

    stop_set = set(stop_names)
    unknown: list[tuple[str, str]] = []

    for a, b in pairing_rules:
        if a not in stop_set:
            unknown.append((a, b))
            assumptions.append(
                f"UNKNOWN STOP IN PAIRING RULE: '{a}' not in stop inventory — rule ({a}, {b}) skipped."
            )
            continue
        if b not in stop_set:
            unknown.append((a, b))
            assumptions.append(
                f"UNKNOWN STOP IN PAIRING RULE: '{b}' not in stop inventory — rule ({a}, {b}) skipped."
            )
            continue
        if b not in adjacency[a]:
            adjacency[a].append(b)
        if a not in adjacency[b]:
            adjacency[b].append(a)

    if unknown:
        assumptions.append(
            f"PAIRING RULES: {len(pairing_rules) - len(unknown)} of {len(pairing_rules)} "
            "explicit pairing rules applied (rest had unknown stop names, see above)."
        )
    else:
        assumptions.append(
            f"PAIRING RULES: All {len(pairing_rules)} explicit pairing rules applied bidirectionally."
        )

    isolated = [s for s, neighbours in adjacency.items() if not neighbours]
    if isolated:
        assumptions.append(
            f"ISOLATED STOPS (no pairing partner): {', '.join(isolated)}. "
            "These are high-value candidates for new pairings — they drive under-minimum bus trips."
        )

    return adjacency, assumptions


def report_isolated_stop_efficiency(
    adjacency: dict[str, list[str]],
    stop_trip_summary: dict[str, dict],
    assumptions: list[str],
) -> list[str]:
    """
    After bus assignment, check whether isolated stops are over-represented
    in under-minimum trips. Returns a list of finding strings.
    """
    findings: list[str] = []
    isolated = {s for s, nbrs in adjacency.items() if not nbrs}
    if not isolated:
        findings.append("All stops have at least one pairing partner.")
        return findings

    total_under = sum(v.get("under_min_trips", 0) for v in stop_trip_summary.values())
    isolated_under = sum(
        stop_trip_summary.get(s, {}).get("under_min_trips", 0) for s in isolated
    )
    if total_under == 0:
        findings.append("No under-minimum trips — isolated stop efficiency not an issue.")
        return findings

    pct = 100 * isolated_under / total_under
    findings.append(
        f"ISOLATED STOP EFFICIENCY: Isolated stops ({', '.join(sorted(isolated))}) "
        f"account for {isolated_under} of {total_under} under-minimum trips ({pct:.1f}%). "
        "Consider adding pairing rules for these stops."
    )
    assumptions.append(findings[-1])
    return findings
