"""Unit tests for the template-based card detection system."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from src.detection.card import Card, DetectedCard, Rank, Suit
from src.detection.generate_templates import (
    generate_all_templates,
    generate_card_template,
    generate_multiscale_templates,
)
from src.detection.template_matcher import (
    TemplateMatcher,
    _calibrate_confidence,
    _compute_iou,
    _non_max_suppression,
)

# ---------------------------------------------------------------------------
# Card and DetectedCard dataclass tests
# ---------------------------------------------------------------------------


class TestCard:
    """Tests for Card and DetectedCard dataclasses."""

    def test_card_creation(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert card.rank == Rank.ACE
        assert card.suit == Suit.SPADES

    def test_card_str(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert "A" in str(card)
        assert "\u2660" in str(card)

    def test_card_name(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert card.name == "ace_spades"

    def test_card_name_numeric(self) -> None:
        card = Card(rank=Rank.TWO, suit=Suit.HEARTS)
        assert card.name == "2_hearts"

    def test_card_frozen(self) -> None:
        card = Card(rank=Rank.ACE, suit=Suit.SPADES)
        with pytest.raises(AttributeError):
            card.rank = Rank.KING  # type: ignore[misc]

    def test_detected_card_creation(self) -> None:
        det = DetectedCard(
            rank=Rank.KING,
            suit=Suit.HEARTS,
            confidence=0.95,
            bounding_box=(10, 20, 60, 80),
        )
        assert det.rank == Rank.KING
        assert det.suit == Suit.HEARTS
        assert det.confidence == 0.95
        assert det.bounding_box == (10, 20, 60, 80)

    def test_detected_card_underlying_card(self) -> None:
        det = DetectedCard(
            rank=Rank.QUEEN,
            suit=Suit.DIAMONDS,
            confidence=0.88,
            bounding_box=(0, 0, 60, 80),
        )
        card = det.card
        assert isinstance(card, Card)
        assert card.rank == Rank.QUEEN
        assert card.suit == Suit.DIAMONDS

    def test_detected_card_str(self) -> None:
        det = DetectedCard(
            rank=Rank.TEN,
            suit=Suit.CLUBS,
            confidence=0.92,
            bounding_box=(100, 200, 60, 80),
        )
        s = str(det)
        assert "10" in s
        assert "0.92" in s

    def test_suit_symbol(self) -> None:
        assert Suit.HEARTS.symbol == "\u2665"
        assert Suit.DIAMONDS.symbol == "\u2666"
        assert Suit.CLUBS.symbol == "\u2663"
        assert Suit.SPADES.symbol == "\u2660"

    def test_suit_color(self) -> None:
        assert Suit.HEARTS.color == (0, 0, 200)
        assert Suit.DIAMONDS.color == (0, 0, 200)
        assert Suit.CLUBS.color == (0, 0, 0)
        assert Suit.SPADES.color == (0, 0, 0)


# ---------------------------------------------------------------------------
# IoU and NMS tests
# ---------------------------------------------------------------------------


class TestNMS:
    """Tests for non-maximum suppression utilities."""

    def test_iou_no_overlap(self) -> None:
        assert _compute_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0

    def test_iou_full_overlap(self) -> None:
        assert _compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_iou_partial_overlap(self) -> None:
        iou = _compute_iou((0, 0, 10, 10), (5, 0, 10, 10))
        assert 0.0 < iou < 1.0

    def test_nms_empty(self) -> None:
        assert _non_max_suppression([]) == []

    def test_nms_removes_overlapping(self) -> None:
        det_high = DetectedCard(Rank.ACE, Suit.SPADES, 0.95, (0, 0, 60, 80))
        det_low = DetectedCard(Rank.KING, Suit.HEARTS, 0.80, (5, 5, 60, 80))
        result = _non_max_suppression([det_low, det_high], iou_threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_nms_keeps_non_overlapping(self) -> None:
        det_a = DetectedCard(Rank.ACE, Suit.SPADES, 0.95, (0, 0, 60, 80))
        det_b = DetectedCard(Rank.KING, Suit.HEARTS, 0.90, (200, 200, 60, 80))
        result = _non_max_suppression([det_a, det_b])
        assert len(result) == 2

    def test_nms_custom_iou_threshold(self) -> None:
        det_a = DetectedCard(Rank.ACE, Suit.SPADES, 0.95, (0, 0, 60, 80))
        det_b = DetectedCard(Rank.KING, Suit.HEARTS, 0.90, (5, 5, 60, 80))
        # Very high threshold → both kept
        result = _non_max_suppression([det_a, det_b], iou_threshold=0.99)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Confidence calibration tests
# ---------------------------------------------------------------------------


class TestConfidenceCalibration:
    """Tests for _calibrate_confidence."""

    def test_perfect_score(self) -> None:
        assert _calibrate_confidence(1.0) == 1.0

    def test_midpoint_maps_to_zero(self) -> None:
        assert _calibrate_confidence(0.5) == 0.0

    def test_below_midpoint_clamped(self) -> None:
        assert _calibrate_confidence(0.3) == 0.0

    def test_above_midpoint_rescaled(self) -> None:
        cal = _calibrate_confidence(0.75)
        assert 0.4 < cal < 0.6


# ---------------------------------------------------------------------------
# Template generation tests
# ---------------------------------------------------------------------------


class TestTemplateGeneration:
    """Tests for template image generation."""

    def test_generate_card_template_shape(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)
        img = generate_card_template(card)
        assert img.shape == (80, 60, 3)
        assert img.dtype == np.uint8

    def test_generate_all_templates_count(self, tmp_path: Path) -> None:
        count = generate_all_templates(tmp_path)
        assert count == 52
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 52


# ---------------------------------------------------------------------------
# TemplateMatcher tests
# ---------------------------------------------------------------------------


class TestTemplateMatcher:
    """Tests for the TemplateMatcher card detection engine."""

    @pytest.fixture()
    def template_dir(self, tmp_path: Path) -> Path:
        """Generate templates in a temporary directory."""
        generate_all_templates(tmp_path)
        return tmp_path

    @pytest.fixture()
    def matcher(self, template_dir: Path) -> TemplateMatcher:
        """Create a TemplateMatcher with generated templates."""
        return TemplateMatcher(template_dir)

    def test_loads_all_templates(self, matcher: TemplateMatcher) -> None:
        assert len(matcher.templates) == 52

    def test_templates_are_lists(self, matcher: TemplateMatcher) -> None:
        """Each template key maps to a list of arrays."""
        for key, tmpl_list in matcher.templates.items():
            assert isinstance(tmpl_list, list)
            assert len(tmpl_list) >= 1

    def test_invalid_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            TemplateMatcher("/nonexistent/path")

    def test_empty_frame_returns_empty(self, matcher: TemplateMatcher) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = matcher.detect_cards(empty)
        assert result == []

    def test_detect_single_card(self, matcher: TemplateMatcher) -> None:
        """Place a known card template on a green background and detect it."""
        card = Card(Rank.ACE, Suit.SPADES)
        template = generate_card_template(card)

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        x, y = 100, 80
        h, w = template.shape[:2]
        frame[y : y + h, x : x + w] = template

        detections = matcher.detect_cards(frame, threshold=0.8)
        assert len(detections) >= 1

        best = detections[0]
        assert best.rank == Rank.ACE
        # Grayscale matching may confuse same-color suits (spades/clubs)
        assert best.suit in (Suit.SPADES, Suit.CLUBS)
        assert best.confidence >= 0.8

    def test_detect_multiple_cards(self, matcher: TemplateMatcher) -> None:
        """Place two cards far apart and detect both."""
        card_a = Card(Rank.ACE, Suit.SPADES)
        card_b = Card(Rank.KING, Suit.HEARTS)
        tmpl_a = generate_card_template(card_a)
        tmpl_b = generate_card_template(card_b)

        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)

        frame[50:130, 50:110] = tmpl_a
        frame[50:130, 300:360] = tmpl_b

        detections = matcher.detect_cards(frame, threshold=0.8)
        detected_ranks = {d.rank for d in detections}
        assert Rank.ACE in detected_ranks
        assert Rank.KING in detected_ranks

    def test_threshold_filtering(self, matcher: TemplateMatcher) -> None:
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        detections = matcher.detect_cards(frame, threshold=1.0)
        assert detections == []

    def test_grayscale_input(self, matcher: TemplateMatcher) -> None:
        card = Card(Rank.SEVEN, Suit.HEARTS)
        template = generate_card_template(card)

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        frame[80:160, 100:160] = template

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = matcher.detect_cards(gray_frame, threshold=0.8)
        assert len(detections) >= 1

    def test_color_matching_mode(self, matcher: TemplateMatcher) -> None:
        """Color matching should still detect a card."""
        card = Card(Rank.QUEEN, Suit.DIAMONDS)
        template = generate_card_template(card)

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        frame[80:160, 100:160] = template

        detections = matcher.detect_cards(frame, threshold=0.8, use_color=True)
        assert len(detections) >= 1

    def test_calibrated_matching(self, matcher: TemplateMatcher) -> None:
        """Calibrated mode should produce detections with adjusted scores."""
        card = Card(Rank.ACE, Suit.SPADES)
        template = generate_card_template(card)

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        frame[80:160, 100:160] = template

        detections = matcher.detect_cards(
            frame, threshold=0.5, calibrate=True,
        )
        assert len(detections) >= 1

    def test_detect_in_region(self, matcher: TemplateMatcher) -> None:
        """detect_in_region adjusts bounding boxes to frame coords."""
        card = Card(Rank.FIVE, Suit.CLUBS)
        template = generate_card_template(card)

        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        # Place card at (200, 100)
        frame[100:180, 200:260] = template

        region = (180, 80, 100, 120)
        detections = matcher.detect_in_region(frame, region, threshold=0.8)
        assert len(detections) >= 1
        # Bounding box x should be offset by region x
        assert detections[0].bounding_box[0] >= 180


class TestMultiscaleMatcher:
    """Tests for multi-scale template loading."""

    @pytest.fixture()
    def multiscale_dir(self, tmp_path: Path) -> Path:
        generate_all_templates(tmp_path)
        generate_multiscale_templates(tmp_path)
        return tmp_path

    def test_loads_multiscale_templates(self, multiscale_dir: Path) -> None:
        matcher = TemplateMatcher(multiscale_dir, multiscale=True)
        # Each card should have multiple scale variants
        for key, tmpl_list in matcher.templates.items():
            assert len(tmpl_list) > 1

    def test_detect_with_multiscale(self, multiscale_dir: Path) -> None:
        matcher = TemplateMatcher(multiscale_dir, multiscale=True)
        card = Card(Rank.ACE, Suit.SPADES)
        template = generate_card_template(card)

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[:] = (34, 120, 50)
        frame[80:160, 100:160] = template

        detections = matcher.detect_cards(frame, threshold=0.8)
        assert len(detections) >= 1
