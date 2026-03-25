"""Tests for the performance measurement harness."""

from __future__ import annotations

import math
import time

import pytest
from src.common.performance import LatencySummary, LatencyTracker, PerfTimer, perf_timer


class TestPerfTimer:
    """Tests for the PerfTimer context manager."""

    def test_measures_elapsed_time(self) -> None:
        with PerfTimer("test_op") as t:
            time.sleep(0.01)  # 10 ms
        assert t.elapsed_ms >= 5  # Allow some slack
        assert t.elapsed_ns > 0

    def test_name_is_stored(self) -> None:
        with PerfTimer("my_operation") as t:
            pass
        assert t.name == "my_operation"

    def test_near_zero_for_noop(self) -> None:
        with PerfTimer("noop") as t:
            pass
        # Should be well under 1 ms for a no-op
        assert t.elapsed_ms < 5

    def test_records_to_tracker(self) -> None:
        tracker = LatencyTracker("test")
        with PerfTimer("op", tracker=tracker):
            time.sleep(0.005)
        assert tracker._count == 1
        assert tracker._samples[0] > 0

    def test_logs_with_logger(self) -> None:
        """Verify that passing a logger does not crash."""
        from src.common.logging import get_logger, reset_logging

        reset_logging()
        logger = get_logger("test.perf")
        with PerfTimer("op", logger=logger):
            pass
        reset_logging()


class TestPerfTimerDecorator:
    """Tests for the perf_timer decorator."""

    def test_decorator_preserves_return_value(self) -> None:
        @perf_timer("add")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_decorator_preserves_function_name(self) -> None:
        @perf_timer("my_func")
        def original() -> None:
            pass

        assert original.__name__ == "original"

    def test_decorator_with_tracker(self) -> None:
        tracker = LatencyTracker("func_tracker")

        @perf_timer("compute", tracker=tracker)
        def compute() -> int:
            return 42

        compute()
        compute()
        assert tracker._count == 2

    def test_decorator_defaults_name_to_qualname(self) -> None:
        tracker = LatencyTracker("test")

        @perf_timer(tracker=tracker)
        def my_function() -> None:
            pass

        my_function()
        assert tracker._count == 1


class TestLatencyTracker:
    """Tests for the LatencyTracker class."""

    def test_record_and_summary(self) -> None:
        tracker = LatencyTracker("test")
        tracker.record(10.0)
        tracker.record(20.0)
        tracker.record(30.0)

        s = tracker.summary()
        assert s.name == "test"
        assert s.count == 3
        assert s.min_ms == 10.0
        assert s.max_ms == 30.0
        assert s.avg_ms == pytest.approx(20.0)

    def test_p95_with_many_samples(self) -> None:
        tracker = LatencyTracker("p95_test")
        for i in range(100):
            tracker.record(float(i))

        s = tracker.summary()
        # p95 of 0..99 should be 94 or 95
        assert s.p95_ms >= 94.0
        assert s.p95_ms <= 95.0

    def test_single_sample(self) -> None:
        tracker = LatencyTracker("single")
        tracker.record(42.0)
        s = tracker.summary()
        assert s.min_ms == 42.0
        assert s.max_ms == 42.0
        assert s.avg_ms == 42.0
        assert s.p95_ms == 42.0
        assert s.count == 1

    def test_summary_raises_when_empty(self) -> None:
        tracker = LatencyTracker("empty")
        with pytest.raises(ValueError, match="No measurements recorded"):
            tracker.summary()

    def test_reset_clears_all(self) -> None:
        tracker = LatencyTracker("reset_test")
        tracker.record(10.0)
        tracker.record(20.0)
        tracker.reset()
        assert tracker._count == 0
        assert len(tracker._samples) == 0
        assert tracker._min == math.inf

    def test_circular_buffer_limits_memory(self) -> None:
        tracker = LatencyTracker("bounded", max_samples=10)
        for i in range(100):
            tracker.record(float(i))
        assert len(tracker._samples) == 10
        assert tracker._count == 100
        # Min/max/avg still track all values
        assert tracker._min == 0.0
        assert tracker._max == 99.0
        assert tracker._sum == pytest.approx(sum(range(100)))


class TestLatencySummary:
    """Tests for the LatencySummary dataclass."""

    def test_fields(self) -> None:
        s = LatencySummary(
            name="op", count=5, min_ms=1.0, max_ms=10.0, avg_ms=5.0, p95_ms=9.5
        )
        assert s.name == "op"
        assert s.count == 5
        assert s.p95_ms == 9.5


class TestOverhead:
    """Verify that the harness adds minimal overhead."""

    def test_timer_overhead_under_1ms(self) -> None:
        """PerfTimer context manager overhead should be well under 1ms."""
        iterations = 1000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            with PerfTimer("noop"):
                pass
        total_ns = time.perf_counter_ns() - start
        per_call_ms = (total_ns / iterations) / 1_000_000
        # Each call should add far less than 1ms of overhead
        assert per_call_ms < 1.0, f"PerfTimer overhead: {per_call_ms:.4f} ms per call"

    def test_tracker_record_overhead_under_1ms(self) -> None:
        tracker = LatencyTracker("overhead_test")
        iterations = 10000
        start = time.perf_counter_ns()
        for _ in range(iterations):
            tracker.record(1.0)
        total_ns = time.perf_counter_ns() - start
        per_call_ms = (total_ns / iterations) / 1_000_000
        assert per_call_ms < 0.1, f"Tracker.record overhead: {per_call_ms:.4f} ms"
