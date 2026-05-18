"""Wave 6 staging load + concurrency validation harness.

Five scenarios exercised against the deployed staging engine:

* ``a`` — Steady ramp 3→100 calls/min over 30 min (ingestion smoothness +
  near-miss autoscaling check).
* ``b`` — Burst 50 calls in 10 s (capacity gate + clean 503s).
* ``c`` — Crash recovery during steady state (ECS stop-task on an active
  task, watch replacement).
* ``d`` — Capacity gate at maxCapacity=5 (deterministic — harness pre-sets
  desiredCount=5 via ECS update-service).
* ``e`` — 4-hour soak at ~50% capacity (memory + FD leak detection,
  vendor steady-state behaviour).

Not in scope (deferred to production rollout): scenario f real-audio
concurrency.

See ``backend/voice-agent/scripts/wave6/README.md`` for the design and
``backend/voice-agent/scripts/staging_load_test.py`` for the entry point.
"""

__all__ = []
