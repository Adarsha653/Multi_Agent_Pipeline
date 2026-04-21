"""Append per-agent wall-clock duration to pipeline state (for API + UI)."""

from __future__ import annotations

import time
from typing import Any


def append_step_duration(state: dict[str, Any], agent_key: str, t_start: float) -> dict[str, Any]:
    dt = round(time.perf_counter() - t_start, 2)
    steps = list(state.get('agent_steps') or [])
    steps.append({'agent': agent_key, 'seconds': dt})
    return {'agent_steps': steps}
