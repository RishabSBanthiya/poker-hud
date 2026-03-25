"""Template-based card detection using OpenCV matchTemplate.

Loads card reference templates from a directory and matches them
against input frames to detect playing cards.  Supports multi-scale
matching, configurable IoU-based NMS, and optional colour matching.
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


def _calibrate_confidence(raw: float, method: int = cv2.TM_CCOEFF_NORMED) -> float:
    """Map a raw matchTemplate score to a calibrated 0-1 confidence.

    For ``TM_CCOEFF_NORMED`` the raw range is already [-1, 1] but in
    practice useful matches fall in [0.5, 1.0].  We apply a simple
    linear rescale so that 0.5 maps to 0.0 and 1.0 stays at 1.0.

    Args:
        raw: Raw correlation score from ``cv2.matchTemplate``.
        method: OpenCV template matching method constant.

    Returns:
        Calibrated confidence clamped to [0.0, 1.0].
    """
    if method == cv2.TM_CCOEFF_NORMED:
        calibrated = max(0.0, (raw - 0.5) / 0.5)
        return min(1.0, calibrated)
    # For other methods just clamp
    return float(np.clip(raw, 0.0, 1.0))


class TemplateMatcher:
    """Card detection engine using OpenCV template matching.

    Supports loading templates from flat or multi-scale directory
    layouts, running matching in grayscale or colour mode, and
    applying configurable non-maximum suppression.

    Attributes:
        templates: Mapping of ``(Rank, Suit)`` to list of grayscale
            template arrays (one per loaded scale).
        color_templates: Mapping of ``(Rank, Suit)`` to list of BGR
            template arrays (populated only when colour matching is used).
    """

    def __init__(
        self,
        template_dir: str | Path,
        *,
        multiscale: bool = False,
    ) -> None:
        """Initialize the matcher by loading templates from a directory.

        Args:
            template_dir: Path to directory containing template images.
                Expected naming: ``{rank}_{suit}.png`` (e.g.,
                ``ace_spades.png``).  When *multiscale* is ``True``,
                subdirectories named by scale (``small/``, ``medium/``,
                ``large/``) are also scanned.
            multiscale: If ``True``, load templates from scale
                subdirectories as well.

        Raises:
            FileNotFoundError: If the template directory does not exist.
        """
        self.template_dir = Path(template_dir)
        if not self.template_dir.is_dir():
            raise FileNotFoundError(
                f"Template directory not found: {self.template_dir}"
            )

        self.templates: dict[tuple[Rank, Suit], list[np.ndarray]] = {}
        self.color_templates: dict[tuple[Rank, Suit], list[np.ndarray]] = {}
        self._multiscale = multiscale
        self._load_templates()

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_templates(self) -> None:
        """Load PNG template images from the template directory."""
        loaded = self._load_from_dir(self.template_dir)

        if self._multiscale:
            for subdir in sorted(self.template_dir.iterdir()):
                if subdir.is_dir():
                    loaded += self._load_from_dir(subdir)

        logger.info(
            "Loaded %d card template images from %s (multiscale=%s)",
            loaded, self.template_dir, self._multiscale,
        )

    def _load_from_dir(self, directory: Path) -> int:
        """Load templates from a single directory.

        Returns:
            Number of templates loaded from this directory.
        """
        loaded = 0
        for path in sorted(directory.glob("*.png")):
            stem = path.stem
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

            color_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if color_img is None:
                logger.warning("Failed to read template image: %s", path)
                continue

            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            key = (rank, suit)
            self.templates.setdefault(key, []).append(gray_img)
            self.color_templates.setdefault(key, []).append(color_img)
            loaded += 1

        return loaded

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_cards(
        self,
        frame: np.ndarray,
        threshold: float = 0.8,
        iou_threshold: float = 0.3,
        *,
        use_color: bool = False,
        calibrate: bool = False,
    ) -> list[DetectedCard]:
        """Detect cards in a frame using template matching.

        When multi-scale templates are loaded each scale variant is
        tried and the best match across scales is kept (NMS removes
        duplicate detections at overlapping positions).

        Args:
            frame: Input image as a BGR (or grayscale) numpy array.
            threshold: Minimum raw confidence score to consider a match
                (0.0-1.0).  When *calibrate* is ``True`` this is
                applied after calibration.
            iou_threshold: IoU threshold for non-maximum suppression.
            use_color: If ``True`` match against colour templates
                (slower but may improve accuracy for suits).
            calibrate: If ``True`` apply confidence calibration to
                raw matchTemplate scores before thresholding.

        Returns:
            List of detected cards sorted by confidence descending.
        """
        if frame.size == 0 or not self.templates:
            return []

        # Prepare frame in the appropriate colour space
        if use_color:
            if len(frame.shape) == 2:
                frame_match = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_match = frame
            template_source = self.color_templates
        else:
            if len(frame.shape) == 3:
                frame_match = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_match = frame
            template_source = self.templates

        detections: list[DetectedCard] = []

        for (rank, suit), tmpl_list in template_source.items():
            for template in tmpl_list:
                new_dets = self._match_single_template(
                    frame_match, template, rank, suit,
                    threshold=threshold,
                    calibrate=calibrate,
                )
                detections.extend(new_dets)

        filtered = _non_max_suppression(detections, iou_threshold=iou_threshold)
        return sorted(filtered, key=lambda d: d.confidence, reverse=True)

    def _match_single_template(
        self,
        frame: np.ndarray,
        template: np.ndarray,
        rank: Rank,
        suit: Suit,
        *,
        threshold: float,
        calibrate: bool,
    ) -> list[DetectedCard]:
        """Run matchTemplate for one template image.

        Args:
            frame: Prepared frame (grayscale or BGR matching template).
            template: Template image array.
            rank: Rank of the card this template represents.
            suit: Suit of the card this template represents.
            threshold: Minimum score to keep.
            calibrate: Whether to apply confidence calibration.

        Returns:
            List of raw (pre-NMS) detections.
        """
        th, tw = template.shape[:2]
        fh, fw = frame.shape[:2]

        if th > fh or tw > fw:
            return []

        method = cv2.TM_CCOEFF_NORMED
        result = cv2.matchTemplate(frame, template, method)

        locations = np.where(result >= (0.5 if calibrate else threshold))
        dets: list[DetectedCard] = []

        for y, x in zip(locations[0], locations[1]):
            raw_score = float(result[y, x])
            if calibrate:
                confidence = _calibrate_confidence(raw_score, method)
            else:
                confidence = raw_score

            if confidence < threshold:
                continue

            dets.append(
                DetectedCard(
                    rank=rank,
                    suit=suit,
                    confidence=confidence,
                    bounding_box=(int(x), int(y), tw, th),
                )
            )

        return dets

    def detect_in_region(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int],
        threshold: float = 0.8,
        iou_threshold: float = 0.3,
        *,
        use_color: bool = False,
        calibrate: bool = False,
    ) -> list[DetectedCard]:
        """Detect cards within a specific region of the frame.

        Crops the frame to *region* before matching, then adjusts
        bounding boxes back to full-frame coordinates.

        Args:
            frame: Full input frame (BGR or grayscale).
            region: ROI as ``(x, y, w, h)`` in frame coordinates.
            threshold: Minimum confidence score.
            iou_threshold: IoU threshold for NMS.
            use_color: Use colour matching.
            calibrate: Apply confidence calibration.

        Returns:
            List of detected cards with frame-relative bounding boxes.
        """
        rx, ry, rw, rh = region
        if len(frame.shape) == 3:
            crop = frame[ry : ry + rh, rx : rx + rw]
        else:
            crop = frame[ry : ry + rh, rx : rx + rw]

        dets = self.detect_cards(
            crop,
            threshold=threshold,
            iou_threshold=iou_threshold,
            use_color=use_color,
            calibrate=calibrate,
        )

        # Offset bounding boxes to full-frame coordinates
        adjusted: list[DetectedCard] = []
        for d in dets:
            bx, by, bw, bh = d.bounding_box
            adjusted.append(
                DetectedCard(
                    rank=d.rank,
                    suit=d.suit,
                    confidence=d.confidence,
                    bounding_box=(bx + rx, by + ry, bw, bh),
                )
            )

        return adjusted
