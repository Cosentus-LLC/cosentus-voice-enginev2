#!/usr/bin/env python3
"""Wave 6 staging load + concurrency validation — entry point.

Usage::

    # Single scenario (smallest, validates harness end-to-end):
    STAGING_API_KEY=<key> uv run python \
      backend/voice-agent/scripts/staging_load_test.py --scenario b

    # Sequential run of a → b → c → d (skip e for now):
    STAGING_API_KEY=<key> uv run python \
      backend/voice-agent/scripts/staging_load_test.py --scenarios a,b,c,d

    # Overnight soak:
    STAGING_API_KEY=<key> nohup uv run python \
      backend/voice-agent/scripts/staging_load_test.py --scenario e \
      > wave6_results/scenario_e.nohup 2>&1 &

    # Full run (everything in order — long):
    STAGING_API_KEY=<key> uv run python \
      backend/voice-agent/scripts/staging_load_test.py --scenarios all

Output lands in ``wave6_results/<UTC-timestamp>/`` (gitignored). At the
end of every invocation the entry point also re-renders
``report.md`` aggregating whatever ``scenario_*.json`` files exist in
that directory.

Shell environment caveat
------------------------

Do NOT ``source backend/voice-agent/scripts/.env.skeleton`` before
running this script. That exports VOICE_API_LAMBDA_NAME and other
engine env vars into your shell, which leak through to CDK if you also
run cdk deploy from the same shell. This entry point only needs
``STAGING_API_KEY`` and standard AWS creds.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Bootstrap so ``from wave6 import ...`` works when the script is run
# directly via ``python staging_load_test.py``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import structlog  # noqa: E402
from wave6 import (  # noqa: E402
    config,
    report,
    scenario_a,
    scenario_b,
    scenario_c,
    scenario_d,
    scenario_e,
)
from wave6.scenario_base import ScenarioResult  # noqa: E402

logger = structlog.get_logger(__name__)


_SCENARIO_RUNNERS = {
    "a": scenario_a.run,
    "b": scenario_b.run,
    "c": scenario_c.run,
    "d": scenario_d.run,
    "e": scenario_e.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave 6 staging load + concurrency validation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        choices=sorted(_SCENARIO_RUNNERS.keys()),
        help="Run a single scenario.",
    )
    group.add_argument(
        "--scenarios",
        help=(
            "Comma-separated list of scenarios to run sequentially, or 'all'. "
            "Examples: 'a,b,c,d', 'all'."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Reuse an existing wave6_results/<ts>/ directory. Useful for "
            "appending the soak after running a-d, or for resuming a "
            "killed soak."
        ),
    )
    return parser.parse_args()


def resolve_scenarios(args: argparse.Namespace) -> list[str]:
    if args.scenario:
        return [args.scenario]
    if args.scenarios:
        if args.scenarios.lower() == "all":
            return list(_SCENARIO_RUNNERS.keys())
        out: list[str] = []
        for name in args.scenarios.split(","):
            name = name.strip()
            if name not in _SCENARIO_RUNNERS:
                raise SystemExit(
                    f"Unknown scenario '{name}'. Choices: {sorted(_SCENARIO_RUNNERS.keys())}"
                )
            out.append(name)
        return out
    raise SystemExit("Must specify --scenario or --scenarios.")


def resolve_paths(args: argparse.Namespace) -> config.RunPaths:
    if args.run_dir is not None:
        return config.RunPaths(root=args.run_dir.resolve())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return config.fresh_run_dir(timestamp)


async def amain(args: argparse.Namespace) -> int:
    # Validate API key early so the script fails fast before any AWS work.
    config.get_api_key()  # raises with a helpful message if missing

    paths = resolve_paths(args)
    scenarios = resolve_scenarios(args)
    logger.info(
        "wave6_run_starting",
        run_dir=str(paths.root),
        scenarios=scenarios,
    )

    failures: list[str] = []
    results: dict[str, ScenarioResult] = {}

    for name in scenarios:
        runner = _SCENARIO_RUNNERS[name]
        try:
            logger.info("scenario_running", scenario=name)
            result = await runner(paths)
            results[name] = result
            logger.info("scenario_done", scenario=name, status=result.overall_status)
            if result.overall_status == "fail":
                failures.append(name)
        except Exception as exc:  # noqa: BLE001 — write what we can
            logger.exception("scenario_crashed", scenario=name, error=str(exc))
            failures.append(name)
            # Keep going — partial results still useful.

    # Always regenerate the aggregate report at the end (covers partials).
    md_path = report.aggregate(paths)
    logger.info(
        "wave6_run_complete",
        run_dir=str(paths.root),
        report=str(md_path),
        failures=failures,
    )

    print()
    print("=" * 72)
    print("Wave 6 run complete.")
    print(f"  Run directory : {paths.root}")
    print(f"  Report        : {md_path}")
    print(f"  Scenarios     : {','.join(scenarios)}")
    print(f"  Failures      : {','.join(failures) if failures else 'none'}")
    print("=" * 72)

    return 1 if failures else 0


def main() -> None:
    args = parse_args()
    rc = asyncio.run(amain(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
