"""Unit tests for the two-head CNN card detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.detection.card import Rank, Suit
from src.detection.cnn_detector import (
    ALL_RANKS,
    ALL_SUITS,
    CNNConfig,
    CNNDetector,
    _conv2d,
    _maxpool2d,
    _preprocess,
    _softmax,
)


class TestPreprocess:
    """Tests for the _preprocess helper."""

    def test_output_shape(self) -> None:
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        out = _preprocess(img, 64)
        assert out.shape == (3, 64, 64)
        assert out.dtype == np.float32

    def test_normalised_range(self) -> None:
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        out = _preprocess(img, 32)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_grayscale_input(self) -> None:
        img = np.zeros((50, 50), dtype=np.uint8)
        out = _preprocess(img, 32)
        assert out.shape == (3, 32, 32)


class TestRankSuitMapping:
    """Tests for rank/suit enum index mapping."""

    def test_rank_count(self) -> None:
        assert len(ALL_RANKS) == 13

    def test_suit_count(self) -> None:
        assert len(ALL_SUITS) == 4

    def test_rank_order_preserved(self) -> None:
        """Enum iteration order matches declaration order."""
        assert ALL_RANKS[0] == Rank.TWO
        assert ALL_RANKS[-1] == Rank.ACE

    def test_suit_order_preserved(self) -> None:
        assert ALL_SUITS[0] == Suit.HEARTS
        assert ALL_SUITS[-1] == Suit.SPADES

    def test_all_ranks_unique(self) -> None:
        assert len(set(ALL_RANKS)) == 13

    def test_all_suits_unique(self) -> None:
        assert len(set(ALL_SUITS)) == 4


class TestConv2d:
    """Tests for the vectorised conv2d helper."""

    def test_output_shape_same_padding(self) -> None:
        x = np.random.randn(3, 8, 8).astype(np.float32)
        w = np.random.randn(4, 3, 3, 3).astype(np.float32)
        b = np.zeros(4, dtype=np.float32)
        out = _conv2d(x, w, b)
        # pad=1 with 3x3 kernel → same spatial size
        assert out.shape == (4, 8, 8)

    def test_bias_applied(self) -> None:
        x = np.zeros((1, 4, 4), dtype=np.float32)
        w = np.zeros((2, 1, 3, 3), dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        out = _conv2d(x, w, b)
        np.testing.assert_allclose(out[0], 1.0)
        np.testing.assert_allclose(out[1], 2.0)


class TestMaxPool2d:
    """Tests for the max pooling helper."""

    def test_output_shape(self) -> None:
        x = np.random.randn(4, 8, 8).astype(np.float32)
        out = _maxpool2d(x, 2)
        assert out.shape == (4, 4, 4)

    def test_max_values(self) -> None:
        x = np.array([[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]], dtype=np.float32)
        out = _maxpool2d(x, 2)
        assert out.shape == (1, 2, 2)
        np.testing.assert_array_equal(out[0], [[6, 8], [14, 16]])


class TestSoftmax:
    """Tests for the softmax helper."""

    def test_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        out = _softmax(x)
        np.testing.assert_allclose(out.sum(), 1.0)

    def test_all_positive(self) -> None:
        x = np.array([-1.0, 0.0, 1.0])
        out = _softmax(x)
        assert (out > 0).all()


class TestCNNDetectorStub:
    """Tests for the CNN detector when no model is loaded."""

    def test_not_ready_without_model(self) -> None:
        cfg = CNNConfig(model_path="/nonexistent/model.npz")
        detector = CNNDetector(cfg)
        assert not detector.is_ready

    def test_returns_empty_without_model(self) -> None:
        cfg = CNNConfig(model_path="/nonexistent/model.npz")
        detector = CNNDetector(cfg)
        frame = np.zeros((100, 80, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []

    def test_custom_config(self) -> None:
        cfg = CNNConfig(
            input_size=32,
            confidence_threshold=0.9,
            model_path="/nonexistent/model.npz",
        )
        detector = CNNDetector(cfg)
        assert detector.config.input_size == 32
        assert detector.config.num_rank_classes == 13
        assert detector.config.num_suit_classes == 4

    def test_empty_frame(self) -> None:
        cfg = CNNConfig(model_path="/nonexistent/model.npz")
        detector = CNNDetector(cfg)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector.detect(empty) == []


def _make_two_head_weights(
    input_size: int = 16,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Create dummy two-head CNN weights matching the architecture.

    Architecture: 3× Conv2d(pad=1) + MaxPool2d(2)
    After 3 pools: spatial = input_size // 8
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pool_size = input_size // 8
    flat_size = 128 * pool_size * pool_size

    return {
        # Backbone (3 input channels for RGB)
        "conv1_w": rng.standard_normal((32, 3, 3, 3)).astype(np.float32) * 0.1,
        "conv1_b": np.zeros(32, dtype=np.float32),
        "conv2_w": rng.standard_normal((64, 32, 3, 3)).astype(np.float32) * 0.1,
        "conv2_b": np.zeros(64, dtype=np.float32),
        "conv3_w": rng.standard_normal((128, 64, 3, 3)).astype(np.float32) * 0.1,
        "conv3_b": np.zeros(128, dtype=np.float32),
        # Rank head
        "rank_fc1_w": rng.standard_normal((flat_size, 128)).astype(np.float32) * 0.1,
        "rank_fc1_b": np.zeros(128, dtype=np.float32),
        "rank_fc2_w": rng.standard_normal((128, 13)).astype(np.float32) * 0.1,
        "rank_fc2_b": np.zeros(13, dtype=np.float32),
        # Suit head
        "suit_fc1_w": rng.standard_normal((flat_size, 64)).astype(np.float32) * 0.1,
        "suit_fc1_b": np.zeros(64, dtype=np.float32),
        "suit_fc2_w": rng.standard_normal((64, 4)).astype(np.float32) * 0.1,
        "suit_fc2_b": np.zeros(4, dtype=np.float32),
    }


class TestCNNDetectorWithWeights:
    """Tests for the CNN detector with a dummy two-head model."""

    @pytest.fixture()
    def detector_with_model(self, tmp_path: Path) -> CNNDetector:
        """Create a detector with random two-head weights."""
        model_path = tmp_path / "test_model.npz"
        weights = _make_two_head_weights(input_size=16)
        np.savez(str(model_path), **weights)

        cfg = CNNConfig(
            model_path=str(model_path),
            input_size=16,
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
        assert isinstance(det.rank, Rank)
        assert isinstance(det.suit, Suit)

    def test_bounding_box_passthrough(self, detector_with_model: CNNDetector) -> None:
        frame = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        bbox = (100, 200, 48, 64)
        result = detector_with_model.detect(frame, bounding_box=bbox)
        assert result[0].bounding_box == bbox

    def test_default_bounding_box(self, detector_with_model: CNNDetector) -> None:
        frame = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        result = detector_with_model.detect(frame)
        assert result[0].bounding_box == (0, 0, 48, 64)

    def test_high_threshold_filters(self, tmp_path: Path) -> None:
        """With threshold=1.0 no detection should pass."""
        model_path = tmp_path / "model.npz"
        weights = _make_two_head_weights(input_size=16)
        np.savez(str(model_path), **weights)

        cfg = CNNConfig(
            model_path=str(model_path),
            input_size=16,
            confidence_threshold=1.0,
        )
        detector = CNNDetector(cfg)
        frame = np.zeros((64, 48, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    def test_forward_returns_two_distributions(
        self, detector_with_model: CNNDetector,
    ) -> None:
        """Forward pass returns rank and suit probability distributions."""
        x = _preprocess(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
            detector_with_model.config.input_size,
        )
        rank_probs, suit_probs = detector_with_model._forward(x)
        assert rank_probs.shape == (13,)
        assert suit_probs.shape == (4,)
        np.testing.assert_allclose(rank_probs.sum(), 1.0, atol=1e-5)
        np.testing.assert_allclose(suit_probs.sum(), 1.0, atol=1e-5)

    def test_bad_weights_not_loaded(self, tmp_path: Path) -> None:
        """Missing keys in NPZ should fail gracefully."""
        model_path = tmp_path / "bad_model.npz"
        np.savez(str(model_path), foo=np.zeros(1))
        cfg = CNNConfig(model_path=str(model_path))
        detector = CNNDetector(cfg)
        assert not detector.is_ready
