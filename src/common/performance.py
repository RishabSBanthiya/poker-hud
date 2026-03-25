"""Lightweight performance measurement harness for poker-hud.

Provides a timer context manager, a decorator, and a latency tracker that
records min/max/avg/p95 statistics. Designed to add less than 1 ms of overhead.

Usage:
    from src.common.performance import PerfTimer, perf_timer, LatencyTracker

    # Context manager
    with PerfTimer("frame_capture") as t:
        frame = capture()
    print(t.elapsed_ms)

    # Decorator
    @perf_timer("detect_cards")
    def detect(frame): ...

    # Tracker for aggregated stats
    tracker = LatencyTracker("capture_pipeline")
    tracker.record(12.3)
    tracker.record(15.1)
    print(tracker.summary())
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, ParamSpec, TypeVar

import structlog

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class LatencySummary:
    """Snapshot of aggregated latency statistics.

    Attributes:
        name: Identifier for the measured operation.
        count: Number of recorded measurements.
        min_ms: Minimum observed latency in milliseconds.
        max_ms: Maximum observed latency in milliseconds.
        avg_ms: Arithmetic mean latency in milliseconds.
        p95_ms: 95th-percentile latency in milliseconds.
    """

    name: str
    count: int
    min_ms: float
    max_ms: float
    avg_ms: float
    p95_ms: float


class LatencyTracker:
    """Accumulates latency measurements and produces summary statistics.

    Keeps a bounded circular buffer to limit memory usage while still
    providing accurate p95 estimates over recent history.

    Args:
        name: Human-readable name for the tracked operation.
        max_samples: Maximum number of samples to retain (default 1000).
    """

    def __init__(self, name: str, max_samples: int = 1000) -> None:
        self.name = name
        self._max_samples = max_samples
        self._samples: list[float] = []
        self._count: int = 0
        self._min: float = math.inf
        self._max: float = -math.inf
        self._sum: float = 0.0

    def record(self, elapsed_ms: float) -> None:
        """Record a single latency measurement.

        Args:
            elapsed_ms: Elapsed time in milliseconds.
        """
        self._count += 1
        self._sum += elapsed_ms
        if elapsed_ms < self._min:
            self._min = elapsed_ms
        if elapsed_ms > self._max:
            self._max = elapsed_ms

        if len(self._samples) < self._max_samples:
            self._samples.append(elapsed_ms)
        else:
            # Overwrite in circular fashion
            idx = (self._count - 1) % self._max_samples
            self._samples[idx] = elapsed_ms

    def summary(self) -> LatencySummary:
        """Compute and return a summary of recorded latencies.

        Returns:
            A ``LatencySummary`` dataclass.

        Raises:
            ValueError: If no measurements have been recorded yet.
        """
        if self._count == 0:
            raise ValueError(f"No measurements recorded for '{self.name}'")

        sorted_samples = sorted(self._samples)
        p95_idx = max(0, int(math.ceil(len(sorted_samples) * 0.95)) - 1)

        return LatencySummary(
            name=self.name,
            count=self._count,
            min_ms=self._min,
            max_ms=self._max,
            avg_ms=self._sum / self._count,
            p95_ms=sorted_samples[p95_idx],
        )

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._samples.clear()
        self._count = 0
        self._min = math.inf
        self._max = -math.inf
        self._sum = 0.0


class PerfTimer:
    """Context manager that measures wall-clock elapsed time.

    Uses ``time.perf_counter_ns`` for nanosecond precision with minimal
    overhead.

    Args:
        name: Label for the timed operation.
        logger: Optional structlog logger. If provided, an info-level
            message is emitted on exit with the elapsed time.
        tracker: Optional ``LatencyTracker`` to automatically record
            each measurement.

    Example::

        with PerfTimer("detect") as t:
            result = detect(frame)
        print(f"Took {t.elapsed_ms:.1f} ms")
    """

    def __init__(
        self,
        name: str,
        *,
        logger: structlog.stdlib.BoundLogger | None = None,
        tracker: LatencyTracker | None = None,
    ) -> None:
        self.name = name
        self._logger = logger
        self._tracker = tracker
        self._start_ns: int = 0
        self.elapsed_ns: int = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> PerfTimer:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ns = time.perf_counter_ns() - self._start_ns
        self.elapsed_ms = self.elapsed_ns / 1_000_000

        if self._tracker is not None:
            self._tracker.record(self.elapsed_ms)

        if self._logger is not None:
            self._logger.info(
                "perf_measurement",
                operation=self.name,
                elapsed_ms=round(self.elapsed_ms, 3),
            )


def perf_timer(
    name: str | None = None,
    *,
    logger: structlog.stdlib.BoundLogger | None = None,
    tracker: LatencyTracker | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that measures and optionally logs function execution time.

    Args:
        name: Operation label. Defaults to the decorated function's
            qualified name.
        logger: Optional structlog logger for automatic logging.
        tracker: Optional ``LatencyTracker`` for recording measurements.

    Returns:
        Decorated function with identical signature.

    Example::

        @perf_timer("card_detect", logger=my_logger)
        def detect_cards(frame):
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        op_name = name if name is not None else func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with PerfTimer(op_name, logger=logger, tracker=tracker):
                return func(*args, **kwargs)

        return wrapper

    return decorator
