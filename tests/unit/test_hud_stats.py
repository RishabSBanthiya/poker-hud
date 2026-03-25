"""Unit tests for HUD stat display widgets."""

from __future__ import annotations

import pytest
from src.common.config import StatsConfig
from src.overlay.hud_stats import (
    HUDStatsWidget,
    SeatPosition,
    StatColor,
    StatsFormatter,
    get_seat_positions,
)
from src.stats.calculations import PlayerStats

# ---------------------------------------------------------------------------
# SeatPosition tests
# ---------------------------------------------------------------------------


class TestSeatPosition:
    """Tests for the SeatPosition dataclass."""

    def test_defaults(self) -> None:
        sp = SeatPosition()
        assert sp.seat_number == 1
        assert sp.x == 0.0
        assert sp.y == 0.0

    def test_custom_values(self) -> None:
        sp = SeatPosition(seat_number=5, x=400.0, y=300.0)
        assert sp.seat_number == 5
        assert sp.x == 400.0


# ---------------------------------------------------------------------------
# get_seat_positions tests
# ---------------------------------------------------------------------------


class TestGetSeatPositions:
    """Tests for seat position generation."""

    def test_two_player_table(self) -> None:
        positions = get_seat_positions(2, 800.0, 600.0)
        assert len(positions) == 2
        assert positions[0].seat_number == 1
        assert positions[1].seat_number == 2

    def test_six_player_table(self) -> None:
        positions = get_seat_positions(6, 800.0, 600.0)
        assert len(positions) == 6

    def test_nine_player_table(self) -> None:
        positions = get_seat_positions(9, 800.0, 600.0)
        assert len(positions) == 9

    def test_ten_player_table(self) -> None:
        positions = get_seat_positions(10, 800.0, 600.0)
        assert len(positions) == 10

    def test_coordinates_scaled_to_window(self) -> None:
        positions = get_seat_positions(2, 1000.0, 800.0)
        # Seat 1 at (0.5, 0.85) should be (500, 680)
        assert positions[0].x == pytest.approx(500.0)
        assert positions[0].y == pytest.approx(680.0)

    def test_invalid_table_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="2-10"):
            get_seat_positions(1, 800.0, 600.0)

    def test_invalid_table_size_too_large(self) -> None:
        with pytest.raises(ValueError, match="2-10"):
            get_seat_positions(11, 800.0, 600.0)

    def test_interpolated_table_size(self) -> None:
        """Table sizes without explicit layouts (e.g. 3, 4, 5, 7, 8)."""
        positions = get_seat_positions(3, 800.0, 600.0)
        assert len(positions) == 3

        positions = get_seat_positions(8, 800.0, 600.0)
        assert len(positions) == 8


# ---------------------------------------------------------------------------
# StatsFormatter tests
# ---------------------------------------------------------------------------


class TestStatsFormatter:
    """Tests for stat value formatting."""

    def setup_method(self) -> None:
        self.formatter = StatsFormatter()
        self.stats = PlayerStats(
            player_name="TestPlayer",
            hands_played=100,
            vpip=22.3,
            pfr=17.8,
            three_bet_pct=8.2,
            aggression_factor=2.15,
        )

    def test_compact_format(self) -> None:
        result = self.formatter.format_compact(self.stats)
        assert result == "22/18/8/2.1"

    def test_detailed_format(self) -> None:
        result = self.formatter.format_detailed(self.stats)
        assert result == "VPIP:22.3% PFR:17.8% 3B:8.2% AF:2.1"

    def test_compact_rounds_correctly(self) -> None:
        stats = PlayerStats(vpip=22.5, pfr=17.4, three_bet_pct=8.6)
        result = self.formatter.format_compact(stats)
        # 22.5 rounds to 22 (banker's rounding), 17.4->17, 8.6->9
        assert result.startswith("22/17/9/")

    def test_zero_stats(self) -> None:
        stats = PlayerStats()
        result = self.formatter.format_compact(stats)
        assert result == "0/0/0/0.0"

    def test_detailed_format_precision(self) -> None:
        stats = PlayerStats(
            vpip=33.33, pfr=25.0, three_bet_pct=12.5, aggression_factor=3.0
        )
        result = self.formatter.format_detailed(stats)
        assert "VPIP:33.3%" in result
        assert "PFR:25.0%" in result


class TestStatsFormatterColors:
    """Tests for stat color coding."""

    def setup_method(self) -> None:
        self.formatter = StatsFormatter()

    def test_normal_vpip_is_green(self) -> None:
        stats = PlayerStats(vpip=22.0)
        assert self.formatter.get_vpip_color(stats) == StatColor.GREEN

    def test_loose_vpip_is_yellow(self) -> None:
        stats = PlayerStats(vpip=27.0)
        assert self.formatter.get_vpip_color(stats) == StatColor.YELLOW

    def test_very_loose_vpip_is_red(self) -> None:
        stats = PlayerStats(vpip=35.0)
        assert self.formatter.get_vpip_color(stats) == StatColor.RED

    def test_very_tight_vpip_is_red(self) -> None:
        stats = PlayerStats(vpip=10.0)
        assert self.formatter.get_vpip_color(stats) == StatColor.RED

    def test_normal_pfr_is_green(self) -> None:
        stats = PlayerStats(pfr=18.0)
        assert self.formatter.get_pfr_color(stats) == StatColor.GREEN

    def test_loose_pfr_is_red(self) -> None:
        stats = PlayerStats(pfr=28.0)
        assert self.formatter.get_pfr_color(stats) == StatColor.RED

    def test_tight_pfr_is_red(self) -> None:
        stats = PlayerStats(pfr=8.0)
        assert self.formatter.get_pfr_color(stats) == StatColor.RED

    def test_overall_color_follows_vpip(self) -> None:
        stats = PlayerStats(vpip=22.0, pfr=18.0)
        assert self.formatter.get_overall_color(stats) == StatColor.GREEN

    def test_custom_thresholds(self) -> None:
        config = StatsConfig(vpip_loose_threshold=40.0, vpip_tight_threshold=10.0)
        formatter = StatsFormatter(config)
        stats = PlayerStats(vpip=35.0)
        # 35 < 40 (loose threshold), > 25 => YELLOW
        assert formatter.get_vpip_color(stats) == StatColor.YELLOW


# ---------------------------------------------------------------------------
# HUDStatsWidget tests
# ---------------------------------------------------------------------------


class TestHUDStatsWidget:
    """Tests for the HUD stats widget."""

    def setup_method(self) -> None:
        self.widget = HUDStatsWidget(
            table_size=6, window_width=800.0, window_height=600.0
        )

    def test_initialization(self) -> None:
        assert self.widget.table_size == 6
        assert len(self.widget.seats) == 6

    def test_update_stats(self) -> None:
        stats = PlayerStats(
            vpip=25.0, pfr=20.0,
            three_bet_pct=10.0, aggression_factor=2.5,
        )
        self.widget.update_stats(1, stats)

        assert self.widget.seats[1].stats is stats
        assert self.widget.get_display_text(1) == "25/20/10/2.5"

    def test_update_stats_invalid_seat(self) -> None:
        stats = PlayerStats()
        with pytest.raises(ValueError, match="Seat 7"):
            self.widget.update_stats(7, stats)

    def test_clear_seat(self) -> None:
        stats = PlayerStats(vpip=25.0)
        self.widget.update_stats(1, stats)
        self.widget.clear_seat(1)

        assert self.widget.seats[1].stats is None
        assert self.widget.get_display_text(1) == ""
        assert self.widget.get_display_color(1) == StatColor.WHITE

    def test_clear_seat_invalid(self) -> None:
        with pytest.raises(ValueError, match="Seat 7"):
            self.widget.clear_seat(7)

    def test_clear_all(self) -> None:
        stats = PlayerStats(vpip=25.0)
        self.widget.update_stats(1, stats)
        self.widget.update_stats(2, stats)
        self.widget.clear_all()

        for seat_num in range(1, 7):
            assert self.widget.seats[seat_num].stats is None

    def test_compact_mode(self) -> None:
        stats = PlayerStats(
            vpip=25.0, pfr=20.0,
            three_bet_pct=10.0, aggression_factor=2.5,
        )
        self.widget.update_stats(1, stats)
        assert "/" in self.widget.get_display_text(1)

    def test_detailed_mode(self) -> None:
        widget = HUDStatsWidget(table_size=2, compact=False)
        stats = PlayerStats(
            vpip=25.0, pfr=20.0,
            three_bet_pct=10.0, aggression_factor=2.5,
        )
        widget.update_stats(1, stats)
        text = widget.get_display_text(1)
        assert "VPIP:" in text
        assert "PFR:" in text

    def test_toggle_compact_reformats(self) -> None:
        stats = PlayerStats(
            vpip=25.0, pfr=20.0,
            three_bet_pct=10.0, aggression_factor=2.5,
        )
        self.widget.update_stats(1, stats)

        self.widget.compact = False
        assert "VPIP:" in self.widget.get_display_text(1)

        self.widget.compact = True
        assert "/" in self.widget.get_display_text(1)

    def test_get_all_display_data(self) -> None:
        stats1 = PlayerStats(vpip=22.0)
        stats2 = PlayerStats(vpip=35.0)
        self.widget.update_stats(1, stats1)
        self.widget.update_stats(3, stats2)

        data = self.widget.get_all_display_data()
        assert len(data) == 2

    def test_get_display_color_empty_seat(self) -> None:
        assert self.widget.get_display_color(1) == StatColor.WHITE

    def test_color_updates_with_stats(self) -> None:
        stats = PlayerStats(vpip=22.0)
        self.widget.update_stats(1, stats)
        assert self.widget.get_display_color(1) == StatColor.GREEN

    def test_get_display_text_invalid_seat(self) -> None:
        assert self.widget.get_display_text(99) == ""

    def test_get_display_color_invalid_seat(self) -> None:
        assert self.widget.get_display_color(99) == StatColor.WHITE
