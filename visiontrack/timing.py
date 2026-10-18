"""Latency bookkeeping with percentile summaries."""

from __future__ import annotations

import resource
import sys
from collections import defaultdict

import numpy as np


def summarize(values: list[float] | np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 3),
        "p50": round(float(np.percentile(v, 50)), 3),
        "p95": round(float(np.percentile(v, 95)), 3),
        "p99": round(float(np.percentile(v, 99)), 3),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
    }


class LatencyRecorder:
    """Collects named per-frame timings in milliseconds."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, ms: float) -> None:
        self.samples[name].append(float(ms))

    def add_many(self, timings: dict[str, float]) -> None:
        for k, v in timings.items():
            self.add(k, v)

    def summary(self) -> dict[str, dict]:
        return {k: summarize(v) for k, v in self.samples.items()}


def peak_rss_mb() -> float:
    """Peak resident set size of this process (MB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024
