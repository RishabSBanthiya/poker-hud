"""Unit tests for enhanced template generation (S2-01)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from src.detection.card import Card, Rank, Suit
from src.detection.generate_templates import (
    SCALES,
    TEMPLATE_HEIGHT,
    TEMPLATE_WIDTH,
    TemplateScale,
    _render_card_at_size,
    generate_all_templates,
    generate_card_template,
    generate_multiscale_templates,
)


class TestRenderCardAtSize:
    """Tests for _render_card_at_size helper."""

    def test_returns_correct_shape(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)
        img = _render_card_at_size(card, 90, 120)
        assert img.shape == (120, 90, 3)
        assert img.dtype == np.uint8

    def test_small_scale(self) -> None:
        card = Card(Rank.TWO, Suit.HEARTS)
        img = _render_card_at_size(card, 40, 54)
        assert img.shape == (54, 40, 3)

    def test_has_white_background(self) -> None:
        card = Card(Rank.FIVE, Suit.CLUBS)
        img = _render_card_at_size(card, 60, 80)
        # A corner pixel (inside border, away from text) should be
        # close to white (may not be pure 255 due to blur)
        corner = img[5, 50]
        assert all(c > 180 for c in corner)


class TestGenerateCardTemplate:
    """Tests for generate_card_template (default medium size)."""

    def test_shape_matches_defaults(self) -> None:
        card = Card(Rank.KING, Suit.DIAMONDS)
        img = generate_card_template(card)
        assert img.shape == (TEMPLATE_HEIGHT, TEMPLATE_WIDTH, 3)

    def test_different_cards_differ(self) -> None:
        img_a = generate_card_template(Card(Rank.ACE, Suit.SPADES))
        img_b = generate_card_template(Card(Rank.TWO, Suit.HEARTS))
        assert not np.array_equal(img_a, img_b)


class TestGenerateAllTemplates:
    """Tests for flat 52-card generation."""

    def test_generates_52_files(self, tmp_path: Path) -> None:
        count = generate_all_templates(tmp_path)
        assert count == 52
        assert len(list(tmp_path.glob("*.png"))) == 52

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "subdir" / "templates"
        generate_all_templates(target)
        assert target.is_dir()

    def test_files_are_readable_images(self, tmp_path: Path) -> None:
        generate_all_templates(tmp_path)
        for png in tmp_path.glob("*.png"):
            img = cv2.imread(str(png))
            assert img is not None
            assert img.shape == (TEMPLATE_HEIGHT, TEMPLATE_WIDTH, 3)


class TestGenerateMultiscaleTemplates:
    """Tests for multi-scale template generation."""

    def test_generates_correct_total(self, tmp_path: Path) -> None:
        count = generate_multiscale_templates(tmp_path)
        assert count == 52 * len(SCALES)

    def test_creates_subdirectories(self, tmp_path: Path) -> None:
        generate_multiscale_templates(tmp_path)
        for scale in SCALES:
            subdir = tmp_path / scale.name
            assert subdir.is_dir()
            assert len(list(subdir.glob("*.png"))) == 52

    def test_custom_scales(self, tmp_path: Path) -> None:
        custom = (TemplateScale("tiny", 20, 28),)
        count = generate_multiscale_templates(tmp_path, scales=custom)
        assert count == 52
        assert (tmp_path / "tiny").is_dir()

    def test_scale_dimensions_are_correct(self, tmp_path: Path) -> None:
        scales = (TemplateScale("test", 45, 60),)
        generate_multiscale_templates(tmp_path, scales=scales)
        sample = next((tmp_path / "test").glob("*.png"))
        img = cv2.imread(str(sample))
        assert img.shape == (60, 45, 3)
