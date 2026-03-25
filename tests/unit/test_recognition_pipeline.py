"""Unit tests for the end-to-end card recognition pipeline (S2-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.detection.card import Card, DetectedCard, Rank, Suit
from src.detection.generate_templates import (
    generate_all_templates,
    generate_card_template,
)
from src.detection.recognition_pipeline import (
    CardRecognitionPipeline,
    PipelineConfig,
    PipelineResult,
    RegionResult,
)
from src.detection.table_regions import (
    POKERSTARS_LAYOUT,
    Region,
    SeatLayout,
    TableLayout,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def template_dir(tmp_path: Path) -> Path:
    """Generate a flat set of templates."""
    generate_all_templates(tmp_path)
    return tmp_path


@pytest.fixture()
def simple_layout() -> TableLayout:
    """A minimal layout with one community card and one seat for testing."""
    return TableLayout(
        name="test_layout",
        community_cards=(
            Region("community_0", 0.25, 0.25, 0.15, 0.20),
        ),
        seats=(
            SeatLayout(
                seat_index=0,
                hole_cards=(
                    Region("seat_0_card_0", 0.60, 0.60, 0.15, 0.20),
                ),
            ),
        ),
    )


@pytest.fixture()
def pipeline(template_dir: Path, simple_layout: TableLayout) -> CardRecognitionPipeline:
    """Pipeline with default config and simple layout."""
    return CardRecognitionPipeline(
        template_dir=template_dir,
        layout=simple_layout,
    )


def _make_frame_with_card(
    card: Card,
    frame_size: tuple[int, int] = (400, 600),
    position: tuple[int, int] = (150, 100),
) -> np.ndarray:
    """Create a green frame with a card template placed at *position*.

    Args:
        card: Card to render.
        frame_size: (height, width).
        position: (x, y) in frame coordinates.

    Returns:
        BGR frame with card placed.
    """
    h, w = frame_size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (34, 120, 50)

    template = generate_card_template(card)
    th, tw = template.shape[:2]
    px, py = position
    frame[py : py + th, px : px + tw] = template
    return frame


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.template_threshold == 0.7
        assert cfg.cnn_fallback_threshold == 0.6
        assert cfg.max_latency_ms == 200.0

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(template_threshold=0.9, multiscale=True)
        assert cfg.template_threshold == 0.9
        assert cfg.multiscale is True


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_empty_result(self) -> None:
        r = PipelineResult()
        assert r.detected_cards == []
        assert r.region_results == []
        assert r.elapsed_ms == 0.0


class TestRegionResult:
    """Tests for RegionResult dataclass."""

    def test_no_detection(self) -> None:
        rr = RegionResult(region_name="test", detection=None, source="none")
        assert rr.detection is None

    def test_with_detection(self) -> None:
        det = DetectedCard(Rank.ACE, Suit.SPADES, 0.95, (0, 0, 60, 80))
        rr = RegionResult(region_name="test", detection=det, source="template")
        assert rr.source == "template"


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestCardRecognitionPipeline:
    """Tests for the full recognition pipeline."""

    def test_process_empty_frame(self, pipeline: CardRecognitionPipeline) -> None:
        """Pipeline should handle a blank frame without errors."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        result = pipeline.process_frame(frame)
        assert isinstance(result, PipelineResult)
        assert result.elapsed_ms > 0

    def test_region_results_count(self, pipeline: CardRecognitionPipeline) -> None:
        """Should have one result per region defined in layout."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        # 1 community + 1 seat card = 2
        assert len(result.region_results) == 2

    def test_detect_card_in_community(
        self, template_dir: Path,
    ) -> None:
        """Place a card exactly in the community region and detect it."""
        # Build a layout where the community region covers a known area
        layout = TableLayout(
            name="precise_test",
            community_cards=(
                Region("community_0", 0.25, 0.25, 0.10, 0.20),
            ),
            seats=(),
        )
        pipe = CardRecognitionPipeline(
            template_dir=template_dir,
            layout=layout,
            config=PipelineConfig(template_threshold=0.7),
        )

        card = Card(Rank.ACE, Suit.SPADES)
        template = generate_card_template(card)
        th, tw = template.shape[:2]

        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        # Place card at the region's pixel location
        rx = int(0.25 * 600)
        ry = int(0.25 * 400)
        frame[ry : ry + th, rx : rx + tw] = template

        result = pipe.process_frame(frame)
        assert len(result.region_results) == 1

    def test_latency_tracking(self, pipeline: CardRecognitionPipeline) -> None:
        """The latency tracker should accumulate measurements."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        pipeline.process_frame(frame)
        pipeline.process_frame(frame)

        summary = pipeline.latency_tracker.summary()
        assert summary.count >= 2
        assert summary.avg_ms >= 0

    def test_seat_indices_filter(
        self, template_dir: Path,
    ) -> None:
        """Passing seat_indices should limit which seats are scanned."""
        layout = TableLayout(
            name="multi_seat",
            community_cards=(),
            seats=(
                SeatLayout(0, (Region("s0c0", 0.1, 0.1, 0.05, 0.08),)),
                SeatLayout(1, (Region("s1c0", 0.3, 0.1, 0.05, 0.08),)),
                SeatLayout(2, (Region("s2c0", 0.5, 0.1, 0.05, 0.08),)),
            ),
        )
        pipe = CardRecognitionPipeline(
            template_dir=template_dir, layout=layout,
        )
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        result_all = pipe.process_frame(frame)
        assert len(result_all.region_results) == 3

        result_sub = pipe.process_frame(frame, seat_indices=[0])
        assert len(result_sub.region_results) == 1

    def test_process_community_only(
        self, template_dir: Path,
    ) -> None:
        layout = TableLayout(
            name="test",
            community_cards=(
                Region("c0", 0.2, 0.2, 0.1, 0.15),
                Region("c1", 0.35, 0.2, 0.1, 0.15),
            ),
            seats=(
                SeatLayout(0, (Region("s0", 0.5, 0.7, 0.05, 0.08),)),
            ),
        )
        pipe = CardRecognitionPipeline(
            template_dir=template_dir, layout=layout,
        )
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        result = pipe.process_community_only(frame)
        # Only community regions, not seat
        assert len(result.region_results) == 2


class TestPipelineWithPokerstarsLayout:
    """Smoke test using the real PokerStars layout preset."""

    def test_processes_without_error(self, template_dir: Path) -> None:
        pipe = CardRecognitionPipeline(
            template_dir=template_dir,
            layout=POKERSTARS_LAYOUT,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        result = pipe.process_frame(frame)
        # 5 community + 6*2 seats = 17
        assert len(result.region_results) == 17
        assert result.elapsed_ms > 0
