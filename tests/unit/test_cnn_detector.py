"""Unit tests for the CNN-based card detector (S2-02)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.detection.card import Rank, Suit
from src.detection.cnn_detector import (
    _ALL_CARDS,
    CNNConfig,
    CNNDetector,
    _index_to_card,
    _preprocess,
)


class TestPreprocess:
    """Tests for the _preprocess helper."""

    def test_output_shape(self) -> None:
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        out = _preprocess(img, 64)
        assert out.shape == (1, 64, 64)
        assert out.dtype == np.float32

    def test_normalised_range(self) -> None:
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        out = _preprocess(img, 32)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_grayscale_input(self) -> None:
        img = np.zeros((50, 50), dtype=np.uint8)
        out = _preprocess(img, 32)
        assert out.shape == (1, 32, 32)


class TestIndexToCard:
    """Tests for the index ↔ card mapping."""

    def test_first_card(self) -> None:
        rank, suit = _index_to_card(0)
        assert isinstance(rank, Rank)
        assert isinstance(suit, Suit)

    def test_all_52_unique(self) -> None:
        assert len(_ALL_CARDS) == 52
        assert len(set(_ALL_CARDS)) == 52

    def test_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            _index_to_card(52)


class TestCNNDetectorStub:
    """Tests for the CNN detector when no model is loaded."""

    def test_not_ready_by_default(self) -> None:
        detector = CNNDetector()
        assert not detector.is_ready

    def test_returns_empty_without_model(self) -> None:
        detector = CNNDetector()
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []

    def test_custom_config(self) -> None:
        cfg = CNNConfig(input_size=32, confidence_threshold=0.9)
        detector = CNNDetector(cfg)
        assert detector.config.input_size == 32

    def test_empty_frame(self) -> None:
        detector = CNNDetector()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector.detect(empty) == []


class TestCNNDetectorWithWeights:
    """Tests for the CNN detector with a dummy model."""

    @pytest.fixture()
    def detector_with_model(self, tmp_path: Path) -> CNNDetector:
        """Create a detector with random weights for testing forward pass."""
        model_path = tmp_path / "test_model.npz"
        num_filters = 4
        kernel_size = 3
        input_size = 16
        # conv output: (num_filters, input_size - kernel_size + 1, ...)
        conv_out_size = input_size - kernel_size + 1
        flat_size = num_filters * conv_out_size * conv_out_size

        rng = np.random.default_rng(42)
        np.savez(
            str(model_path),
            conv1_w=rng.standard_normal(
                (num_filters, 1, kernel_size, kernel_size),
            ).astype(np.float32),
            conv1_b=np.zeros(num_filters, dtype=np.float32),
            fc_w=rng.standard_normal((flat_size, 52)).astype(np.float32),
            fc_b=np.zeros(52, dtype=np.float32),
        )

        cfg = CNNConfig(
            model_path=str(model_path),
            input_size=input_size,
            confidence_threshold=0.01,
        )
        return CNNDetector(cfg)

    def test_is_ready(self, detector_with_model: CNNDetector) -> None:
        assert detector_with_model.is_ready

    def test_returns_detection(self, detector_with_model: CNNDetector) -> None:
        frame = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        result = detector_with_model.detect(frame)
        assert len(result) == 1
        det = result[0]
        assert 0.0 <= det.confidence <= 1.0

    def test_bounding_box_passthrough(self, detector_with_model: CNNDetector) -> None:
        frame = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        bbox = (100, 200, 48, 64)
        result = detector_with_model.detect(frame, bounding_box=bbox)
        assert result[0].bounding_box == bbox

    def test_high_threshold_filters(self, tmp_path: Path) -> None:
        """With threshold=1.0 no detection should pass."""
        model_path = tmp_path / "model.npz"
        rng = np.random.default_rng(0)
        np.savez(
            str(model_path),
            conv1_w=rng.standard_normal((4, 1, 3, 3)).astype(np.float32),
            conv1_b=np.zeros(4, dtype=np.float32),
            fc_w=rng.standard_normal((4 * 14 * 14, 52)).astype(np.float32),
            fc_b=np.zeros(52, dtype=np.float32),
        )
        cfg = CNNConfig(
            model_path=str(model_path),
            input_size=16,
            confidence_threshold=1.0,
        )
        detector = CNNDetector(cfg)
        frame = np.zeros((64, 48, 3), dtype=np.uint8)
        assert detector.detect(frame) == []
