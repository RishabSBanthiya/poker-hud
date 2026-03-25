"""Template-based card detection using OpenCV matchTemplate.

Loads card reference templates from a directory and matches them
against input frames to detect playing cards.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.detection.card import DetectedCard, Rank, Suit

logger = logging.getLogger(__name__)

# Mapping from filename components to enums
_RANK_MAP: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "10": Rank.TEN,
    "jack": Rank.JACK,
    "queen": Rank.QUEEN,
    "king": Rank.KING,
    "ace": Rank.ACE,
}

_SUIT_MAP: dict[str, Suit] = {
    "hearts": Suit.HEARTS,
    "diamonds": Suit.DIAMONDS,
    "clubs": Suit.CLUBS,
    "spades": Suit.SPADES,
}


def _compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box_a: First box as (x, y, w, h).
        box_b: Second box as (x, y, w, h).

    Returns:
        IoU value between 0.0 and 1.0.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x_left = max(ax, bx)
    y_top = max(ay, by)
    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def _non_max_suppression(
    detections: list[DetectedCard],
    iou_threshold: float = 0.3,
) -> list[DetectedCard]:
    """Filter overlapping detections, keeping the highest-confidence one.

    Args:
        detections: List of detected cards to filter.
        iou_threshold: IoU threshold above which detections are suppressed.

    Returns:
        Filtered list of detected cards.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[DetectedCard] = []

    for det in sorted_dets:
        suppress = False
        for kept_det in kept:
            if _compute_iou(det.bounding_box, kept_det.bounding_box) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    return kept


class TemplateMatcher:
    """Card detection engine using OpenCV template matching.

    Loads card templates from a directory and matches them against
    input frames to detect playing cards.

    Attributes:
        templates: Dictionary mapping (Rank, Suit) to grayscale template arrays.
    """

    def __init__(self, template_dir: str | Path) -> None:
        """Initialize the matcher by loading templates from a directory.

        Args:
            template_dir: Path to directory containing template images.
                Expected naming: ``{rank}_{suit}.png`` (e.g., ``ace_spades.png``).

        Raises:
            FileNotFoundError: If the template directory does not exist.
        """
        self.template_dir = Path(template_dir)
        if not self.template_dir.is_dir():
            raise FileNotFoundError(
                f"Template directory not found: {self.template_dir}"
            )

        self.templates: dict[tuple[Rank, Suit], np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all PNG template images from the template directory."""
        loaded = 0
        for path in sorted(self.template_dir.glob("*.png")):
            stem = path.stem  # e.g., "ace_spades"
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                logger.warning("Skipping unrecognized template file: %s", path.name)
                continue

            rank_str, suit_str = parts
            rank = _RANK_MAP.get(rank_str)
            suit = _SUIT_MAP.get(suit_str)

            if rank is None or suit is None:
                logger.warning("Skipping unrecognized template file: %s", path.name)
                continue

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Failed to read template image: %s", path)
                continue

            self.templates[(rank, suit)] = img
            loaded += 1

        logger.info(
            "Loaded %d card templates from %s", loaded, self.template_dir
        )

    def detect_cards(
        self,
        frame: np.ndarray,
        threshold: float = 0.8,
        iou_threshold: float = 0.3,
    ) -> list[DetectedCard]:
        """Detect cards in a BGR frame using template matching.

        Args:
            frame: Input image as a BGR numpy array.
            threshold: Minimum confidence score to consider a match (0.0-1.0).
            iou_threshold: IoU threshold for non-maximum suppression.

        Returns:
            List of detected cards sorted by confidence descending.
        """
        if frame.size == 0 or not self.templates:
            return []

        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        detections: list[DetectedCard] = []

        for (rank, suit), template in self.templates.items():
            th, tw = template.shape[:2]

            # Skip if template is larger than the frame
            if th > frame_gray.shape[0] or tw > frame_gray.shape[1]:
                continue

            result = cv2.matchTemplate(
                frame_gray, template, cv2.TM_CCOEFF_NORMED
            )

            locations = np.where(result >= threshold)

            for y, x in zip(locations[0], locations[1]):
                confidence = float(result[y, x])
                detections.append(
                    DetectedCard(
                        rank=rank,
                        suit=suit,
                        confidence=confidence,
                        bounding_box=(int(x), int(y), tw, th),
                    )
                )

        filtered = _non_max_suppression(detections, iou_threshold=iou_threshold)
        return sorted(filtered, key=lambda d: d.confidence, reverse=True)
