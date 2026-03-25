"""Poker table layout regions and card-area extraction.

Defines configurable ``TableLayout`` presets for popular poker clients
(PokerStars, 888poker, generic) and a ``RegionLocalizer`` that crops
card-relevant sub-images from a captured frame.

Usage:
    from src.detection.table_regions import (
        RegionLocalizer, TableLayout, POKERSTARS_LAYOUT,
    )

    localizer = RegionLocalizer(POKERSTARS_LAYOUT)
    regions = localizer.extract_regions(frame)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """A rectangular region on the poker table.

    Coordinates are expressed as fractions of the full frame size
    (0.0–1.0) so that a single layout works across resolutions.

    Attributes:
        name: Human-readable label (e.g. ``"community_0"``).
        x_frac: Left edge as fraction of frame width.
        y_frac: Top edge as fraction of frame height.
        w_frac: Width as fraction of frame width.
        h_frac: Height as fraction of frame height.
    """

    name: str
    x_frac: float
    y_frac: float
    w_frac: float
    h_frac: float

    def to_pixel_box(
        self, frame_width: int, frame_height: int,
    ) -> tuple[int, int, int, int]:
        """Convert fractional coordinates to pixel ``(x, y, w, h)``.

        Args:
            frame_width: Width of the source frame in pixels.
            frame_height: Height of the source frame in pixels.

        Returns:
            Bounding box in pixel coordinates.
        """
        x = int(round(self.x_frac * frame_width))
        y = int(round(self.y_frac * frame_height))
        w = max(1, int(round(self.w_frac * frame_width)))
        h = max(1, int(round(self.h_frac * frame_height)))
        # Clamp to frame bounds
        x = min(x, frame_width - 1)
        y = min(y, frame_height - 1)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)
        return (x, y, w, h)


@dataclass(frozen=True)
class SeatLayout:
    """Card regions for a single player seat.

    Attributes:
        seat_index: Zero-based seat number.
        hole_cards: Regions for the player's hole cards (typically 2).
    """

    seat_index: int
    hole_cards: tuple[Region, ...]


@dataclass(frozen=True)
class TableLayout:
    """Full poker table layout defining where cards appear.

    Attributes:
        name: Layout identifier (e.g. ``"pokerstars_6max"``).
        community_cards: Regions for the five community cards.
        seats: Per-seat hole card regions.
        dealer_button: Optional region for the dealer button.
        table_width: Reference width the fractional coords were
            designed for (informational only).
        table_height: Reference height the fractional coords were
            designed for (informational only).
    """

    name: str
    community_cards: tuple[Region, ...]
    seats: tuple[SeatLayout, ...]
    dealer_button: Region | None = None
    table_width: int = 1920
    table_height: int = 1080


@dataclass(frozen=True)
class ExtractedRegion:
    """A cropped sub-image extracted from a frame.

    Attributes:
        name: The source ``Region.name``.
        image: Cropped BGR numpy array.
        pixel_box: ``(x, y, w, h)`` in full-frame coordinates.
    """

    name: str
    image: np.ndarray
    pixel_box: tuple[int, int, int, int]


# ------------------------------------------------------------------
# Built-in layout presets
# ------------------------------------------------------------------

def _community_regions() -> tuple[Region, ...]:
    """Five community card regions centred horizontally."""
    card_w = 0.045
    card_h = 0.075
    y = 0.38
    start_x = 0.34
    gap = 0.055
    return tuple(
        Region(
            name=f"community_{i}",
            x_frac=start_x + i * gap,
            y_frac=y,
            w_frac=card_w,
            h_frac=card_h,
        )
        for i in range(5)
    )


def _seat_regions_6max() -> tuple[SeatLayout, ...]:
    """Hole-card regions for a 6-max table."""
    card_w = 0.035
    card_h = 0.065
    gap = 0.04

    # Approximate seat positions (x_center, y_center)
    positions = [
        (0.50, 0.82),   # seat 0 — hero (bottom centre)
        (0.15, 0.65),   # seat 1 — left
        (0.15, 0.25),   # seat 2 — top-left
        (0.50, 0.10),   # seat 3 — top centre
        (0.80, 0.25),   # seat 4 — top-right
        (0.80, 0.65),   # seat 5 — right
    ]

    seats: list[SeatLayout] = []
    for idx, (cx, cy) in enumerate(positions):
        cards = tuple(
            Region(
                name=f"seat_{idx}_card_{c}",
                x_frac=cx - card_w + c * gap,
                y_frac=cy - card_h / 2,
                w_frac=card_w,
                h_frac=card_h,
            )
            for c in range(2)
        )
        seats.append(SeatLayout(seat_index=idx, hole_cards=cards))

    return tuple(seats)


def _seat_regions_9max() -> tuple[SeatLayout, ...]:
    """Hole-card regions for a 9-max (full ring) table."""
    card_w = 0.035
    card_h = 0.065
    gap = 0.04

    positions = [
        (0.50, 0.85),   # seat 0 — hero
        (0.22, 0.78),   # seat 1
        (0.10, 0.55),   # seat 2
        (0.15, 0.25),   # seat 3
        (0.35, 0.10),   # seat 4
        (0.60, 0.10),   # seat 5
        (0.80, 0.25),   # seat 6
        (0.88, 0.55),   # seat 7
        (0.75, 0.78),   # seat 8
    ]

    seats: list[SeatLayout] = []
    for idx, (cx, cy) in enumerate(positions):
        cards = tuple(
            Region(
                name=f"seat_{idx}_card_{c}",
                x_frac=cx - card_w + c * gap,
                y_frac=cy - card_h / 2,
                w_frac=card_w,
                h_frac=card_h,
            )
            for c in range(2)
        )
        seats.append(SeatLayout(seat_index=idx, hole_cards=cards))

    return tuple(seats)


POKERSTARS_LAYOUT = TableLayout(
    name="pokerstars_6max",
    community_cards=_community_regions(),
    seats=_seat_regions_6max(),
    dealer_button=Region("dealer_button", 0.44, 0.68, 0.025, 0.035),
)

EIGHT88_LAYOUT = TableLayout(
    name="888poker_6max",
    community_cards=_community_regions(),
    seats=_seat_regions_6max(),
    dealer_button=Region("dealer_button", 0.44, 0.68, 0.025, 0.035),
)

GENERIC_9MAX_LAYOUT = TableLayout(
    name="generic_9max",
    community_cards=_community_regions(),
    seats=_seat_regions_9max(),
    dealer_button=Region("dealer_button", 0.44, 0.70, 0.025, 0.035),
)


# ------------------------------------------------------------------
# Region localizer
# ------------------------------------------------------------------

class RegionLocalizer:
    """Extracts card sub-images from a frame based on a table layout.

    Args:
        layout: The ``TableLayout`` defining where cards are expected.
    """

    def __init__(self, layout: TableLayout) -> None:
        self.layout = layout

    def extract_community_regions(
        self, frame: np.ndarray,
    ) -> list[ExtractedRegion]:
        """Crop community card regions from *frame*.

        Args:
            frame: Full table frame (BGR numpy array).

        Returns:
            List of ``ExtractedRegion`` for each community card slot.
        """
        fh, fw = frame.shape[:2]
        results: list[ExtractedRegion] = []
        for region in self.layout.community_cards:
            extracted = self._crop_region(frame, region, fw, fh)
            if extracted is not None:
                results.append(extracted)
        return results

    def extract_seat_regions(
        self,
        frame: np.ndarray,
        seat_indices: list[int] | None = None,
    ) -> list[ExtractedRegion]:
        """Crop hole-card regions for one or more seats.

        Args:
            frame: Full table frame (BGR numpy array).
            seat_indices: Seats to extract.  ``None`` means all seats.

        Returns:
            List of ``ExtractedRegion`` for each hole card slot.
        """
        fh, fw = frame.shape[:2]
        results: list[ExtractedRegion] = []

        for seat in self.layout.seats:
            if seat_indices is not None and seat.seat_index not in seat_indices:
                continue
            for region in seat.hole_cards:
                extracted = self._crop_region(frame, region, fw, fh)
                if extracted is not None:
                    results.append(extracted)

        return results

    def extract_all_regions(
        self, frame: np.ndarray,
    ) -> list[ExtractedRegion]:
        """Crop all card regions (community + all seats) from *frame*.

        Args:
            frame: Full table frame.

        Returns:
            Combined list of extracted regions.
        """
        return self.extract_community_regions(frame) + self.extract_seat_regions(frame)

    def extract_dealer_button(
        self, frame: np.ndarray,
    ) -> ExtractedRegion | None:
        """Crop the dealer button region.

        Args:
            frame: Full table frame.

        Returns:
            Extracted region or ``None`` if no dealer button is defined.
        """
        if self.layout.dealer_button is None:
            return None
        fh, fw = frame.shape[:2]
        return self._crop_region(frame, self.layout.dealer_button, fw, fh)

    @staticmethod
    def _crop_region(
        frame: np.ndarray,
        region: Region,
        frame_width: int,
        frame_height: int,
    ) -> ExtractedRegion | None:
        """Crop a single region from the frame.

        Returns ``None`` if the resulting crop would be empty.
        """
        x, y, w, h = region.to_pixel_box(frame_width, frame_height)
        if w <= 0 or h <= 0:
            return None

        if len(frame.shape) == 3:
            crop = frame[y : y + h, x : x + w].copy()
        else:
            crop = frame[y : y + h, x : x + w].copy()

        if crop.size == 0:
            return None

        return ExtractedRegion(
            name=region.name,
            image=crop,
            pixel_box=(x, y, w, h),
        )
