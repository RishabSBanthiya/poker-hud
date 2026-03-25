"""Unit tests for table region localization (S2-03)."""

from __future__ import annotations

import numpy as np
import pytest
from src.detection.table_regions import (
    EIGHT88_LAYOUT,
    GENERIC_9MAX_LAYOUT,
    POKERSTARS_LAYOUT,
    ExtractedRegion,
    Region,
    RegionLocalizer,
    SeatLayout,
    TableLayout,
)

# ---------------------------------------------------------------------------
# Region dataclass tests
# ---------------------------------------------------------------------------


class TestRegion:
    """Tests for the Region dataclass."""

    def test_to_pixel_box(self) -> None:
        r = Region("test", x_frac=0.5, y_frac=0.25, w_frac=0.1, h_frac=0.2)
        box = r.to_pixel_box(1000, 800)
        assert box == (500, 200, 100, 160)

    def test_clamps_to_frame(self) -> None:
        r = Region("edge", x_frac=0.99, y_frac=0.99, w_frac=0.1, h_frac=0.1)
        x, y, w, h = r.to_pixel_box(100, 100)
        assert x + w <= 100
        assert y + h <= 100

    def test_minimum_dimensions(self) -> None:
        r = Region("tiny", x_frac=0.0, y_frac=0.0, w_frac=0.001, h_frac=0.001)
        _, _, w, h = r.to_pixel_box(100, 100)
        assert w >= 1
        assert h >= 1


class TestSeatLayout:
    """Tests for the SeatLayout dataclass."""

    def test_creation(self) -> None:
        cards = (
            Region("card_0", 0.1, 0.1, 0.05, 0.08),
            Region("card_1", 0.16, 0.1, 0.05, 0.08),
        )
        seat = SeatLayout(seat_index=0, hole_cards=cards)
        assert seat.seat_index == 0
        assert len(seat.hole_cards) == 2


# ---------------------------------------------------------------------------
# Built-in layout tests
# ---------------------------------------------------------------------------


class TestBuiltinLayouts:
    """Tests for pre-defined table layouts."""

    def test_pokerstars_community_count(self) -> None:
        assert len(POKERSTARS_LAYOUT.community_cards) == 5

    def test_pokerstars_seat_count(self) -> None:
        assert len(POKERSTARS_LAYOUT.seats) == 6

    def test_pokerstars_has_dealer_button(self) -> None:
        assert POKERSTARS_LAYOUT.dealer_button is not None

    def test_888_layout(self) -> None:
        assert len(EIGHT88_LAYOUT.seats) == 6

    def test_generic_9max(self) -> None:
        assert len(GENERIC_9MAX_LAYOUT.seats) == 9
        assert len(GENERIC_9MAX_LAYOUT.community_cards) == 5

    def test_all_regions_in_bounds(self) -> None:
        """All fractional coords should be in [0, 1]."""
        for layout in (POKERSTARS_LAYOUT, EIGHT88_LAYOUT, GENERIC_9MAX_LAYOUT):
            for region in layout.community_cards:
                assert 0.0 <= region.x_frac <= 1.0
                assert 0.0 <= region.y_frac <= 1.0
            for seat in layout.seats:
                for region in seat.hole_cards:
                    assert 0.0 <= region.x_frac <= 1.0
                    assert 0.0 <= region.y_frac <= 1.0


# ---------------------------------------------------------------------------
# RegionLocalizer tests
# ---------------------------------------------------------------------------


class TestRegionLocalizer:
    """Tests for the RegionLocalizer class."""

    @pytest.fixture()
    def localizer(self) -> RegionLocalizer:
        return RegionLocalizer(POKERSTARS_LAYOUT)

    @pytest.fixture()
    def frame(self) -> np.ndarray:
        """A 1920x1080 green frame."""
        f = np.zeros((1080, 1920, 3), dtype=np.uint8)
        f[:] = (34, 120, 50)
        return f

    def test_extract_community_regions(
        self, localizer: RegionLocalizer, frame: np.ndarray,
    ) -> None:
        regions = localizer.extract_community_regions(frame)
        assert len(regions) == 5
        for r in regions:
            assert isinstance(r, ExtractedRegion)
            assert r.image.size > 0
            assert "community" in r.name

    def test_extract_seat_regions_all(
        self, localizer: RegionLocalizer, frame: np.ndarray,
    ) -> None:
        regions = localizer.extract_seat_regions(frame)
        # 6 seats × 2 hole cards
        assert len(regions) == 12

    def test_extract_seat_regions_subset(
        self, localizer: RegionLocalizer, frame: np.ndarray,
    ) -> None:
        regions = localizer.extract_seat_regions(frame, seat_indices=[0, 1])
        assert len(regions) == 4  # 2 seats × 2 cards

    def test_extract_all_regions(
        self, localizer: RegionLocalizer, frame: np.ndarray,
    ) -> None:
        regions = localizer.extract_all_regions(frame)
        assert len(regions) == 5 + 12  # community + seats

    def test_extract_dealer_button(
        self, localizer: RegionLocalizer, frame: np.ndarray,
    ) -> None:
        result = localizer.extract_dealer_button(frame)
        assert result is not None
        assert result.name == "dealer_button"

    def test_no_dealer_button(self, frame: np.ndarray) -> None:
        layout = TableLayout(
            name="no_button",
            community_cards=(),
            seats=(),
            dealer_button=None,
        )
        loc = RegionLocalizer(layout)
        assert loc.extract_dealer_button(frame) is None

    def test_pixel_box_coordinates(
        self, localizer: RegionLocalizer, frame: np.ndarray,
    ) -> None:
        regions = localizer.extract_community_regions(frame)
        for r in regions:
            x, y, w, h = r.pixel_box
            assert x >= 0
            assert y >= 0
            assert w > 0
            assert h > 0
            assert x + w <= 1920
            assert y + h <= 1080

    def test_smaller_frame(self, localizer: RegionLocalizer) -> None:
        """Layout should work on any resolution."""
        small_frame = np.zeros((540, 960, 3), dtype=np.uint8)
        regions = localizer.extract_community_regions(small_frame)
        assert len(regions) == 5

    def test_grayscale_frame(self, localizer: RegionLocalizer) -> None:
        gray = np.zeros((1080, 1920), dtype=np.uint8)
        regions = localizer.extract_community_regions(gray)
        assert len(regions) == 5


class TestCustomLayout:
    """Tests for user-defined custom layouts."""

    def test_custom_layout(self) -> None:
        layout = TableLayout(
            name="custom_2seat",
            community_cards=(
                Region("comm_0", 0.4, 0.4, 0.05, 0.08),
            ),
            seats=(
                SeatLayout(
                    seat_index=0,
                    hole_cards=(
                        Region("s0_c0", 0.2, 0.8, 0.04, 0.06),
                        Region("s0_c1", 0.25, 0.8, 0.04, 0.06),
                    ),
                ),
            ),
        )
        localizer = RegionLocalizer(layout)
        frame = np.zeros((600, 800, 3), dtype=np.uint8)

        comm = localizer.extract_community_regions(frame)
        assert len(comm) == 1

        seats = localizer.extract_seat_regions(frame)
        assert len(seats) == 2
