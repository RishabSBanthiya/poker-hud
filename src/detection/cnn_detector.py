"""CNN-based card detection fallback.

Provides a ``CNNDetector`` class that implements the same detection
interface as ``TemplateMatcher`` but uses a convolutional neural
network for classification.  The initial implementation is a stub
that returns empty results until a trained model is available.

The CNN architecture is defined purely in terms of numpy operations
to avoid adding PyTorch as a hard dependency.  A PyTorch training
harness can be added later under ``src/detection/train_cnn.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import DetectedCard, Rank, Suit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CNNConfig:
    """Configuration for the CNN card detector.

    Attributes:
        model_path: Filesystem path to a saved model weights file.
        input_size: Expected square input dimension in pixels.
        confidence_threshold: Minimum score to emit a detection.
        num_classes: Number of output classes (52 cards).
    """

    model_path: str | Path = "models/card_detector.npz"
    input_size: int = 64
    confidence_threshold: float = 0.5
    num_classes: int = 52


# Ordered list of all 52 cards for mapping class indices ↔ (Rank, Suit)
_ALL_CARDS: list[tuple[Rank, Suit]] = [
    (rank, suit) for suit in Suit for rank in Rank
]


def _index_to_card(index: int) -> tuple[Rank, Suit]:
    """Map a class index (0-51) to a ``(Rank, Suit)`` pair.

    Args:
        index: Class index.

    Returns:
        Corresponding (Rank, Suit) tuple.

    Raises:
        IndexError: If index is out of range.
    """
    return _ALL_CARDS[index]


def _preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Resize and normalise an image for CNN input.

    Args:
        image: BGR or grayscale numpy array.
        size: Target square dimension.

    Returns:
        Normalised float32 array of shape ``(1, size, size)`` (grayscale).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return normalised.reshape(1, size, size)


class CNNDetector:
    """Card detector backed by a simple CNN.

    Until a trained model is loaded the detector returns empty results.
    This allows the recognition pipeline to fall through gracefully
    while the CNN is being developed.

    Example::

        detector = CNNDetector()
        cards = detector.detect(frame)
    """

    def __init__(self, config: CNNConfig | None = None) -> None:
        """Initialise the CNN detector.

        Args:
            config: Optional configuration overrides.
        """
        self.config = config or CNNConfig()
        self._weights_loaded = False
        self._try_load_weights()

    def _try_load_weights(self) -> None:
        """Attempt to load model weights from disk.

        Fails silently so the detector can be used as a no-op stub
        before a model is trained.
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
            self._conv1_w = data["conv1_w"]
            self._conv1_b = data["conv1_b"]
            self._fc_w = data["fc_w"]
            self._fc_b = data["fc_b"]
            self._weights_loaded = True
            logger.info("Loaded CNN weights from %s", path)
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
        """Detect cards in *frame* using the CNN.

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
        probabilities = self._forward(preprocessed)

        best_idx = int(np.argmax(probabilities))
        best_prob = float(probabilities[best_idx])

        if best_prob < self.config.confidence_threshold:
            return []

        rank, suit = _index_to_card(best_idx)

        bx, by, bw, bh = bounding_box
        if bw == 0 or bh == 0:
            h, w = frame.shape[:2]
            bounding_box = (0, 0, w, h)

        return [
            DetectedCard(
                rank=rank,
                suit=suit,
                confidence=best_prob,
                bounding_box=bounding_box,
            )
        ]

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Run a simple forward pass (conv → relu → flatten → fc → softmax).

        This is a minimal numpy-only inference path.  A real deployment
        would use PyTorch or ONNX Runtime.

        Args:
            x: Preprocessed input of shape ``(1, H, W)``.

        Returns:
            Probability vector of shape ``(num_classes,)``.
        """
        # Conv layer (valid cross-correlation, single filter bank)
        # conv1_w shape: (num_filters, 1, kH, kW)
        num_filters = self._conv1_w.shape[0]
        kh, kw = self._conv1_w.shape[2], self._conv1_w.shape[3]
        _, ih, iw = x.shape
        oh, ow = ih - kh + 1, iw - kw + 1

        conv_out = np.zeros((num_filters, oh, ow), dtype=np.float32)
        for f in range(num_filters):
            for i in range(oh):
                for j in range(ow):
                    patch = x[0, i : i + kh, j : j + kw]
                    w = self._conv1_w[f, 0]
                    conv_out[f, i, j] = (
                        np.sum(patch * w) + self._conv1_b[f]
                    )

        # ReLU
        conv_out = np.maximum(conv_out, 0)

        # Flatten
        flat = conv_out.reshape(-1)

        # FC layer
        logits = flat @ self._fc_w + self._fc_b

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
