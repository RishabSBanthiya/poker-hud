"""Unit tests for the dataset management module (S2-01)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.detection.card import Card, Rank, Suit
from src.detection.dataset import (
    TemplateDataset,
    TemplateMetadata,
)
from src.detection.generate_templates import (
    generate_all_templates,
    generate_multiscale_templates,
)


@pytest.fixture()
def flat_dataset_dir(tmp_path: Path) -> Path:
    """Generate a flat set of 52 templates."""
    generate_all_templates(tmp_path)
    return tmp_path


@pytest.fixture()
def multiscale_dataset_dir(tmp_path: Path) -> Path:
    """Generate multi-scale templates plus flat defaults."""
    generate_all_templates(tmp_path)
    generate_multiscale_templates(tmp_path)
    return tmp_path


class TestTemplateMetadata:
    """Tests for the TemplateMetadata dataclass."""

    def test_creation(self) -> None:
        m = TemplateMetadata(
            rank="ace", suit="spades", scale="medium",
            source="synthetic", width=60, height=80,
            filename="ace_spades.png",
        )
        assert m.rank == "ace"
        assert m.filename == "ace_spades.png"

    def test_frozen(self) -> None:
        m = TemplateMetadata(
            rank="2", suit="hearts", scale="small",
            source="synthetic", width=40, height=54,
            filename="2_hearts.png",
        )
        with pytest.raises(AttributeError):
            m.rank = "3"  # type: ignore[misc]


class TestTemplateDatasetFlat:
    """Tests for loading a flat directory layout."""

    def test_loads_all_52(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        assert ds.card_count() == 52
        assert len(ds) == 52

    def test_get_templates_for_card(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        card = Card(Rank.ACE, Suit.SPADES)
        entries = ds.get_templates_for_card(card)
        assert len(entries) >= 1
        assert entries[0].metadata.rank == "ace"
        assert entries[0].metadata.suit == "spades"

    def test_entry_image_is_valid(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        entry = ds.entries[0]
        assert isinstance(entry.image, np.ndarray)
        assert entry.image.ndim == 3

    def test_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            TemplateDataset.from_directory("/nonexistent/path")

    def test_iteration(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        count = sum(1 for _ in ds)
        assert count == 52


class TestTemplateDatasetMultiscale:
    """Tests for loading a multi-scale directory layout."""

    def test_loads_multiscale_entries(self, multiscale_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(multiscale_dataset_dir)
        # 52 flat + 52*3 multiscale = 208
        assert len(ds) == 52 + 52 * 3

    def test_get_templates_at_scale(self, multiscale_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(multiscale_dataset_dir)
        small = ds.get_templates_at_scale("small")
        assert len(small) == 52


class TestAddCapturedImage:
    """Tests for adding real captured images."""

    def test_add_and_persist(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        original_len = len(ds)

        card = Card(Rank.ACE, Suit.SPADES)
        fake_img = np.zeros((80, 60, 3), dtype=np.uint8)
        entry = ds.add_captured_image(card, fake_img)

        assert len(ds) == original_len + 1
        assert entry.metadata.source == "captured"
        assert (flat_dataset_dir / "captured").is_dir()

    def test_captured_entry_has_correct_metadata(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        card = Card(Rank.TEN, Suit.HEARTS)
        img = np.ones((100, 70, 3), dtype=np.uint8) * 128
        entry = ds.add_captured_image(card, img)
        assert entry.metadata.width == 70
        assert entry.metadata.height == 100
        assert entry.metadata.scale == "captured"


class TestMetadataPersistence:
    """Tests for saving/loading metadata JSON."""

    def test_save_and_load(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        meta_path = ds.save_metadata()
        assert meta_path.exists()

        loaded = TemplateDataset.load_metadata(meta_path)
        assert len(loaded) == 52
        assert all(isinstance(m, TemplateMetadata) for m in loaded)

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            TemplateDataset.load_metadata("/nonexistent/metadata.json")

    def test_json_is_valid(self, flat_dataset_dir: Path) -> None:
        ds = TemplateDataset.from_directory(flat_dataset_dir)
        meta_path = ds.save_metadata()
        data = json.loads(meta_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 52
        assert "rank" in data[0]
