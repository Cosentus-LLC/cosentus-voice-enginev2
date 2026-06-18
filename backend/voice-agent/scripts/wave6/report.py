"""Aggregate per-scenario JSON files into a single markdown report.

Run once at the end of a Wave 6 session. Reads every
``scenario_<x>.json`` in the run dir and produces ``report.md`` with:

* Top-level pass/fail/inconclusive count.
* Per-scenario table: status, duration, headline numbers.
* Per-scenario detail blocks with each Check listed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config

_SCENARIOS_IN_ORDER = ("a", "b", "c", "d", "e")


# ── Symbols (no emojis to keep the report grep-friendly) ────────────────────

_STATUS_TEXT = {
    "pass": "PASS",
    "fail": "FAIL",
    "inconclusive": "INCONCLUSIVE",
}


def aggregate(paths: config.RunPaths) -> Path:
    """Read all available scenario_*.json files and write report.md."""
    results: dict[str, dict[str, Any]] = {}
    for name in _SCENARIOS_IN_ORDER:
        json_path = paths.scenario_json(name)
        if json_path.exists():
            try:
                results[name] = json.loads(json_path.read_text())
            except json.JSONDecodeError as exc:
                results[name] = {
                    "scenario": name,
                    "overall_status": "fail",
                    "_parse_error": str(exc),
                }

    md_lines: list[str] = []
    md_lines.append("# Wave 6 — staging load + concurrency validation")
    md_lines.append("")
    md_lines.append(f"Run directory: `{paths.root}`")
    md_lines.append("")
    md_lines.append(_render_summary_table(results))
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    for name in _SCENARIOS_IN_ORDER:
        if name not in results:
            md_lines.append(f"## Scenario {name.upper()} — NOT RUN")
            md_lines.append("")
            md_lines.append(f"`{paths.scenario_json(name)}` not found.")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            continue
        md_lines.extend(_render_scenario_detail(name, results[name]))
        md_lines.append("---")
        md_lines.append("")

    paths.report_md.write_text("\n".join(md_lines))
    return paths.report_md


# ── Rendering helpers ───────────────────────────────────────────────────────


def _render_summary_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "## Overall summary",
        "",
        "| Scenario | Status | Duration (s) | Calls | Accepted | Rejected 503 | Other | "
        "P95 latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in _SCENARIOS_IN_ORDER:
        r = results.get(name)
        if r is None:
            lines.append(f"| {name.upper()} | NOT RUN | — | — | — | — | — | — |")
            continue
        status = _STATUS_TEXT.get(r.get("overall_status", "inconclusive"), "INCONCLUSIVE")
        dur = r.get("duration_secs", "—")
        calls = r.get("calls") or {}
        count = calls.get("count", "—")
        accepted = calls.get("accepted_202", "—")
        rejected = calls.get("rejected_503", "—")
        other = sum((calls.get("other") or {}).values()) if calls.get("other") else 0
        latency = (calls.get("latency_ms") or {}).get("p95", "—")
        lines.append(
            f"| {name.upper()} | {status} | {dur} | {count} | {accepted} | "
            f"{rejected} | {other} | {latency} |"
        )
    return "\n".join(lines)


def _render_scenario_detail(name: str, result: dict[str, Any]) -> list[str]:
    desc = result.get("description") or "(no description)"
    status = _STATUS_TEXT.get(result.get("overall_status", "inconclusive"), "INCONCLUSIVE")
    started = result.get("started_at", "—")
    ended = result.get("ended_at", "—")
    duration = result.get("duration_secs", "—")

    lines = [
        f"## Scenario {name.upper()} — {status}",
        "",
        f"**Description.** {desc}",
        "",
        f"**Window.** {started} → {ended} ({duration} s).",
        "",
    ]

    cfg = result.get("config") or {}
    if cfg:
        lines.append("**Scenario knobs.**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(cfg, indent=2, default=str))
        lines.append("```")
        lines.append("")

    calls = result.get("calls") or {}
    if calls:
        lines.append("**/start outcomes.**")
        lines.append("")
        lines.append(_render_kv_table(calls))
        lines.append("")

    cw = result.get("cloudwatch") or {}
    if cw:
        lines.append("**CloudWatch stats.**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(cw, indent=2, default=str))
        lines.append("```")
        lines.append("")

    ecs = result.get("ecs") or {}
    if ecs:
        lines.append("**ECS service state.**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(ecs, indent=2, default=str))
        lines.append("```")
        lines.append("")

    checks = result.get("checks") or []
    if checks:
        lines.append("**Checks.**")
        lines.append("")
        lines.append("| Check | Status | Observed | Expected | Note |")
        lines.append("|---|---|---|---|---|")
        for c in checks:
            check_status = _STATUS_TEXT.get(c.get("status", "inconclusive"), "INCONCLUSIVE")
            obs = _short(c.get("observed"))
            exp = _short(c.get("expected"))
            note = (c.get("note") or "").replace("|", "\\|")
            lines.append(
                f"| {c.get('name', '?')} — {c.get('description', '')} "
                f"| {check_status} | `{obs}` | `{exp}` | {note} |"
            )
        lines.append("")

    notes = result.get("notes") or []
    if notes:
        lines.append("**Notes.**")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    return lines


def _render_kv_table(d: dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        rows.append(f"| `{k}` | `{_short(v)}` |")
    return "\n".join(["| Key | Value |", "|---|---|"] + rows)


def _short(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"), default=str)
    s = str(v)
    return s if len(s) < 80 else s[:77] + "..."
