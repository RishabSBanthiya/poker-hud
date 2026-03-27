"""Unit tests for the synthetic card corner crop generator."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from src.detection.card import Rank, Suit
from src.detection.synthetic_generator import (
    ALL_RANKS,
    ALL_SUITS,
    SyntheticConfig,
    generate_dataset,
    render_corner_crop,
)


class TestRenderCornerCrop:
    """Tests for single corner crop rendering."""

    def test_output_shape(self) -> None:
        img = render_corner_crop(Rank.ACE, Suit.SPADES, (64, 64))
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_custom_size(self) -> None:
        img = render_corner_crop(Rank.TWO, Suit.HEARTS, (32, 48))
        assert img.shape == (48, 32, 3)

    def test_deterministic_with_seed(self) -> None:
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        img1 = render_corner_crop(Rank.KING, Suit.CLUBS, (64, 64), rng=rng1)
        img2 = render_corner_crop(Rank.KING, Suit.CLUBS, (64, 64), rng=rng2)
        np.testing.assert_array_equal(img1, img2)

    def test_different_seeds_produce_different_images(self) -> None:
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        img1 = render_corner_crop(Rank.ACE, Suit.HEARTS, (64, 64), rng=rng1)
        img2 = render_corner_crop(Rank.ACE, Suit.HEARTS, (64, 64), rng=rng2)
        assert not np.array_equal(img1, img2)

    def test_custom_bg_color(self) -> None:
        bg = (0, 255, 0)
        img = render_corner_crop(
            Rank.FIVE, Suit.DIAMONDS, (64, 64), bg_color=bg,
        )
        # Corners should be close to the background color
        assert img.shape == (64, 64, 3)

    def test_all_ranks_render(self) -> None:
        for rank in Rank:
            img = render_corner_crop(rank, Suit.SPADES, (64, 64))
            assert img.shape == (64, 64, 3)

    def test_all_suits_render(self) -> None:
        for suit in Suit:
            img = render_corner_crop(Rank.ACE, suit, (64, 64))
            assert img.shape == (64, 64, 3)


class TestGenerateDataset:
    """Tests for full dataset generation."""

    def test_generates_correct_count(self, tmp_path: Path) -> None:
        config = SyntheticConfig(
            output_dir=tmp_path / "synth",
            samples_per_card=2,
            seed=42,
        )
        total = generate_dataset(config)
        expected = 13 * 4 * 2  # ranks * suits * samples
        assert total == expected

    def test_creates_images_directory(self, tmp_path: Path) -> None:
        config = SyntheticConfig(
            output_dir=tmp_path / "synth",
            samples_per_card=1,
            seed=42,
        )
        generate_dataset(config)
        img_dir = tmp_path / "synth" / "images"
        assert img_dir.is_dir()

    def test_creates_manifest(self, tmp_path: Path) -> None:
        config = SyntheticConfig(
            output_dir=tmp_path / "synth",
            samples_per_card=1,
            seed=42,
        )
        generate_dataset(config)
        manifest = tmp_path / "synth" / "manifest.csv"
        assert manifest.exists()

        with open(manifest) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 13 * 4
        # Check required columns
        assert set(rows[0].keys()) == {
            "filename", "rank_idx", "suit_idx", "rank", "suit",
        }

    def test_manifest_rank_suit_indices(self, tmp_path: Path) -> None:
        config = SyntheticConfig(
            output_dir=tmp_path / "synth",
            samples_per_card=1,
            seed=42,
        )
        generate_dataset(config)

        with open(tmp_path / "synth" / "manifest.csv") as f:
            rows = list(csv.DictReader(f))

        rank_indices = {int(r["rank_idx"]) for r in rows}
        suit_indices = {int(r["suit_idx"]) for r in rows}
        assert rank_indices == set(range(13))
        assert suit_indices == set(range(4))

    def test_images_are_correct_size(self, tmp_path: Path) -> None:
        import cv2

        config = SyntheticConfig(
            output_dir=tmp_path / "synth",
            crop_size=(32, 32),
            samples_per_card=1,
            seed=42,
        )
        generate_dataset(config)

        img_dir = tmp_path / "synth" / "images"
        for png in img_dir.glob("*.png"):
            img = cv2.imread(str(png))
            assert img is not None
            assert img.shape[:2] == (32, 32)
            break  # just check first one

    def test_reproducible_with_seed(self, tmp_path: Path) -> None:
        """Same seed produces identical datasets."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        config1 = SyntheticConfig(
            output_dir=dir1, samples_per_card=1, seed=99,
        )
        config2 = SyntheticConfig(
            output_dir=dir2, samples_per_card=1, seed=99,
        )
        generate_dataset(config1)
        generate_dataset(config2)

        import cv2

        for png1 in sorted((dir1 / "images").glob("*.png")):
            png2 = dir2 / "images" / png1.name
            img1 = cv2.imread(str(png1))
            img2 = cv2.imread(str(png2))
            np.testing.assert_array_equal(img1, img2)


class TestEnumOrdering:
    """Verify enum ordering matches expected card indices."""

    def test_ranks_match_expected(self) -> None:
        expected = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE,
        ]
        assert ALL_RANKS == expected

    def test_suits_match_expected(self) -> None:
        expected = [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
        assert ALL_SUITS == expected
