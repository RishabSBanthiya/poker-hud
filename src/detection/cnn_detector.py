"""CNN-based card detection with two-head rank+suit architecture.

Provides a ``CNNDetector`` class that classifies card corner crops
into separate rank (13 classes) and suit (4 classes) predictions
using a shared convolutional backbone.

The CNN architecture is implemented purely in numpy for inference,
avoiding a hard PyTorch dependency.  The corresponding PyTorch
training harness lives in ``src/detection/train_cnn.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import DetectedCard, Rank, Suit

logger = logging.getLogger(__name__)

# Ordered lists matching enum declaration order for index mapping.
ALL_RANKS: list[Rank] = list(Rank)
ALL_SUITS: list[Suit] = list(Suit)


@dataclass(frozen=True)
class CNNConfig:
    """Configuration for the two-head CNN card detector.

    Attributes:
        model_path: Filesystem path to a saved model weights file.
        input_size: Expected square input dimension in pixels.
        confidence_threshold: Minimum combined score to emit a detection.
        num_rank_classes: Number of rank output classes (13).
        num_suit_classes: Number of suit output classes (4).
    """

    model_path: str | Path = "models/card_detector.npz"
    input_size: int = 64
    confidence_threshold: float = 0.5
    num_rank_classes: int = 13
    num_suit_classes: int = 4


def _preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Resize and normalise an image for CNN input.

    Uses 3-channel RGB so the model can leverage colour to distinguish
    suits (red vs black).

    Args:
        image: BGR or grayscale numpy array.
        size: Target square dimension.

    Returns:
        Normalised float32 array of shape ``(3, size, size)``.
    """
    if len(image.shape) == 2:
        # Grayscale → BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    # HWC → CHW
    return normalised.transpose(2, 0, 1)


def _conv2d(
    x: np.ndarray, w: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Vectorised 2-D convolution with same-size padding.

    Args:
        x: Input of shape ``(C_in, H, W)``.
        w: Filters of shape ``(C_out, C_in, kH, kW)``.
        b: Bias of shape ``(C_out,)``.

    Returns:
        Output of shape ``(C_out, H, W)`` (same spatial size).
    """
    c_out, c_in, kh, kw = w.shape
    _, ih, iw = x.shape

    # Pad input for same-size output
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    x_padded = np.pad(
        x,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )

    # Build im2col matrix: extract all patches as rows
    # Output shape: (ih * iw, c_in * kh * kw)
    from numpy.lib.stride_tricks import as_strided

    _, ph, pw = x_padded.shape
    # Strides for (C_in, H_out, W_out, kH, kW)
    s_c, s_h, s_w = x_padded.strides
    patches = as_strided(
        x_padded,
        shape=(c_in, ih, iw, kh, kw),
        strides=(s_c, s_h, s_w, s_h, s_w),
    )
    # (ih, iw, c_in, kh, kw) → (ih*iw, c_in*kh*kw)
    col = patches.transpose(1, 2, 0, 3, 4).reshape(ih * iw, c_in * kh * kw)

    # filters: (c_out, c_in*kh*kw) → transpose for matmul
    filters_flat = w.reshape(c_out, c_in * kh * kw).T

    out = col @ filters_flat + b  # (ih*iw, c_out)
    return out.T.reshape(c_out, ih, iw)


def _maxpool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    """2×2 max pooling with stride equal to pool size.

    Args:
        x: Input of shape ``(C, H, W)``.
        size: Pool window size.

    Returns:
        Pooled output of shape ``(C, H//size, W//size)``.
    """
    c, h, w = x.shape
    oh, ow = h // size, w // size
    # Truncate to exact multiples
    x = x[:, : oh * size, : ow * size]
    return x.reshape(c, oh, size, ow, size).max(axis=(2, 4))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


class CNNDetector:
    """Card detector backed by a two-head rank+suit CNN.

    The model uses a shared convolutional backbone with separate
    fully-connected heads for rank (13 classes) and suit (4 classes)
    classification.

    Until a trained model is loaded the detector returns empty results.

    Example::

        detector = CNNDetector()
        cards = detector.detect(frame)
    """

    def __init__(self, config: CNNConfig | None = None) -> None:
        self.config = config or CNNConfig()
        self._weights_loaded = False
        self._try_load_weights()

    def _try_load_weights(self) -> None:
        """Attempt to load model weights from disk.

        Supports the two-head NPZ format with keys:
        conv{1,2,3}_{w,b}, rank_fc{1,2}_{w,b}, suit_fc{1,2}_{w,b}.
        """
        path = Path(self.config.model_path)
        if not path.exists():
            logger.info(
                "CNN model not found at %s — detector will return empty results",
                path,
            )
            return

        try:
            data = np.load(str(path), allow_pickle=False)

            # Backbone conv layers (BN fused at export time)
            self._conv1_w = data["conv1_w"]
            self._conv1_b = data["conv1_b"]
            self._conv2_w = data["conv2_w"]
            self._conv2_b = data["conv2_b"]
            self._conv3_w = data["conv3_w"]
            self._conv3_b = data["conv3_b"]

            # Rank head
            self._rank_fc1_w = data["rank_fc1_w"]
            self._rank_fc1_b = data["rank_fc1_b"]
            self._rank_fc2_w = data["rank_fc2_w"]
            self._rank_fc2_b = data["rank_fc2_b"]

            # Suit head
            self._suit_fc1_w = data["suit_fc1_w"]
            self._suit_fc1_b = data["suit_fc1_b"]
            self._suit_fc2_w = data["suit_fc2_w"]
            self._suit_fc2_b = data["suit_fc2_b"]

            self._weights_loaded = True
            logger.info("Loaded two-head CNN weights from %s", path)
        except (KeyError, ValueError) as exc:
            logger.warning("Failed to load CNN weights from %s: %s", path, exc)

    @property
    def is_ready(self) -> bool:
        """Return ``True`` if a trained model has been loaded."""
        return self._weights_loaded

    def detect(
        self,
        frame: np.ndarray,
        bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> list[DetectedCard]:
        """Detect a card in *frame* using the two-head CNN.

        Args:
            frame: BGR or grayscale image (full frame or cropped region).
            bounding_box: If the frame is a cropped region, pass the
                region's ``(x, y, w, h)`` in full-frame coordinates so
                the returned detections have correct positions.

        Returns:
            List of detected cards.  Empty when no model is loaded or
            confidence is below threshold.
        """
        if not self._weights_loaded:
            return []

        if frame.size == 0:
            return []

        preprocessed = _preprocess(frame, self.config.input_size)
        rank_probs, suit_probs = self._forward(preprocessed)

        rank_idx = int(np.argmax(rank_probs))
        suit_idx = int(np.argmax(suit_probs))
        rank_conf = float(rank_probs[rank_idx])
        suit_conf = float(suit_probs[suit_idx])

        combined_confidence = rank_conf * suit_conf

        if combined_confidence < self.config.confidence_threshold:
            return []

        rank = ALL_RANKS[rank_idx]
        suit = ALL_SUITS[suit_idx]

        bx, by, bw, bh = bounding_box
        if bw == 0 or bh == 0:
            h, w = frame.shape[:2]
            bounding_box = (0, 0, w, h)

        return [
            DetectedCard(
                rank=rank,
                suit=suit,
                confidence=combined_confidence,
                bounding_box=bounding_box,
            )
        ]

    def _forward(
        self, x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the two-head forward pass.

        Architecture: 3× (Conv3×3 → ReLU → MaxPool2×2) → flatten
        → rank head (FC → ReLU → FC → softmax)
        → suit head (FC → ReLU → FC → softmax)

        Args:
            x: Preprocessed input of shape ``(1, H, W)``.

        Returns:
            Tuple of (rank_probabilities, suit_probabilities).
        """
        # Backbone: 3 conv blocks
        x = _maxpool2d(np.maximum(_conv2d(x, self._conv1_w, self._conv1_b), 0))
        x = _maxpool2d(np.maximum(_conv2d(x, self._conv2_w, self._conv2_b), 0))
        x = _maxpool2d(np.maximum(_conv2d(x, self._conv3_w, self._conv3_b), 0))

        flat = x.reshape(-1)

        # Rank head: FC → ReLU → FC → softmax
        rank_h = np.maximum(flat @ self._rank_fc1_w + self._rank_fc1_b, 0)
        rank_logits = rank_h @ self._rank_fc2_w + self._rank_fc2_b

        # Suit head: FC → ReLU → FC → softmax
        suit_h = np.maximum(flat @ self._suit_fc1_w + self._suit_fc1_b, 0)
        suit_logits = suit_h @ self._suit_fc2_w + self._suit_fc2_b

        return _softmax(rank_logits), _softmax(suit_logits)
