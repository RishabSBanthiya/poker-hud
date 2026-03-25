"""HUD stat display widgets for per-player statistics.

Provides widgets to display player statistics (VPIP, PFR, 3-Bet, AF)
near each player's seat position on the poker table, with color coding
based on configurable thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from src.common.config import StatsConfig
from src.stats.calculations import PlayerStats

logger = logging.getLogger(__name__)


class StatColor(Enum):
    """Color categories for stat display.

    Values are RGBA tuples (0.0-1.0) for use with AppKit colors.
    """

    GREEN = (0.0, 0.8, 0.0, 1.0)   # Normal / TAG
    YELLOW = (1.0, 0.8, 0.0, 1.0)  # Loose / LAG
    RED = (1.0, 0.2, 0.2, 1.0)     # Extreme (very loose or very tight)
    WHITE = (1.0, 1.0, 1.0, 1.0)   # Default / neutral


@dataclass
class SeatPosition:
    """Screen coordinates for a player seat on the poker table.

    Attributes:
        seat_number: Seat number (1-based, 1 to max_seats).
        x: Horizontal position in points relative to poker window.
        y: Vertical position in points relative to poker window.
    """

    seat_number: int = 1
    x: float = 0.0
    y: float = 0.0


# Default seat layouts for common table sizes.
# Coordinates are normalized fractions of the poker window (0.0-1.0).
_DEFAULT_SEAT_LAYOUTS: dict[int, list[tuple[float, float]]] = {
    2: [
        (0.5, 0.85),   # Seat 1 (hero, bottom center)
        (0.5, 0.15),   # Seat 2 (villain, top center)
    ],
    6: [
        (0.5, 0.85),   # Seat 1 (bottom center)
        (0.15, 0.65),  # Seat 2 (left middle-low)
        (0.15, 0.35),  # Seat 3 (left middle-high)
        (0.5, 0.15),   # Seat 4 (top center)
        (0.85, 0.35),  # Seat 5 (right middle-high)
        (0.85, 0.65),  # Seat 6 (right middle-low)
    ],
    9: [
        (0.5, 0.85),   # Seat 1
        (0.2, 0.75),   # Seat 2
        (0.1, 0.5),    # Seat 3
        (0.2, 0.25),   # Seat 4
        (0.4, 0.15),   # Seat 5
        (0.6, 0.15),   # Seat 6
        (0.8, 0.25),   # Seat 7
        (0.9, 0.5),    # Seat 8
        (0.8, 0.75),   # Seat 9
    ],
    10: [
        (0.5, 0.85),   # Seat 1
        (0.2, 0.8),    # Seat 2
        (0.1, 0.6),    # Seat 3
        (0.1, 0.4),    # Seat 4
        (0.2, 0.2),    # Seat 5
        (0.5, 0.15),   # Seat 6
        (0.8, 0.2),    # Seat 7
        (0.9, 0.4),    # Seat 8
        (0.9, 0.6),    # Seat 9
        (0.8, 0.8),    # Seat 10
    ],
}


def get_seat_positions(
    table_size: int,
    window_width: float,
    window_height: float,
) -> list[SeatPosition]:
    """Generate seat positions for a given table size and window dimensions.

    Args:
        table_size: Number of seats at the table (2-10).
        window_width: Poker window width in points.
        window_height: Poker window height in points.

    Returns:
        List of SeatPosition objects with absolute coordinates.

    Raises:
        ValueError: If table_size is not between 2 and 10.
    """
    if not 2 <= table_size <= 10:
        raise ValueError(f"Table size must be 2-10, got {table_size}")

    # Find closest layout
    if table_size in _DEFAULT_SEAT_LAYOUTS:
        layout = _DEFAULT_SEAT_LAYOUTS[table_size]
    else:
        # Interpolate: use the next larger layout, trimmed
        for size in sorted(_DEFAULT_SEAT_LAYOUTS.keys()):
            if size >= table_size:
                layout = _DEFAULT_SEAT_LAYOUTS[size][:table_size]
                break
        else:
            layout = _DEFAULT_SEAT_LAYOUTS[10][:table_size]

    return [
        SeatPosition(
            seat_number=i + 1,
            x=frac_x * window_width,
            y=frac_y * window_height,
        )
        for i, (frac_x, frac_y) in enumerate(layout)
    ]


class StatsFormatter:
    """Formats player statistics for display in the HUD.

    Supports compact and detailed display modes with color coding
    based on configurable thresholds.

    Args:
        config: Stats display threshold configuration.
    """

    def __init__(self, config: StatsConfig | None = None) -> None:
        self._config = config or StatsConfig()

    @property
    def config(self) -> StatsConfig:
        """Return the current stats configuration."""
        return self._config

    def format_compact(self, stats: PlayerStats) -> str:
        """Format stats in compact mode: "22/18/8/2.1".

        Shows VPIP/PFR/3Bet/AF in a single line.

        Args:
            stats: Player statistics to format.

        Returns:
            Compact formatted string.
        """
        vpip = self._format_pct(stats.vpip, decimals=0)
        pfr = self._format_pct(stats.pfr, decimals=0)
        three_bet = self._format_pct(stats.three_bet_pct, decimals=0)
        af = self._format_af(stats.aggression_factor)
        return f"{vpip}/{pfr}/{three_bet}/{af}"

    def format_detailed(self, stats: PlayerStats) -> str:
        """Format stats in detailed mode with labels.

        Shows "VPIP:22% PFR:18% 3B:8% AF:2.1".

        Args:
            stats: Player statistics to format.

        Returns:
            Detailed formatted string.
        """
        vpip = self._format_pct(stats.vpip, decimals=1)
        pfr = self._format_pct(stats.pfr, decimals=1)
        three_bet = self._format_pct(stats.three_bet_pct, decimals=1)
        af = self._format_af(stats.aggression_factor)
        return f"VPIP:{vpip}% PFR:{pfr}% 3B:{three_bet}% AF:{af}"

    def get_vpip_color(self, stats: PlayerStats) -> StatColor:
        """Determine color for VPIP display.

        Args:
            stats: Player statistics.

        Returns:
            Color category based on VPIP thresholds.
        """
        if stats.vpip > self._config.vpip_loose_threshold:
            return StatColor.RED
        if stats.vpip < self._config.vpip_tight_threshold:
            return StatColor.RED
        if stats.vpip > 25.0:
            return StatColor.YELLOW
        return StatColor.GREEN

    def get_pfr_color(self, stats: PlayerStats) -> StatColor:
        """Determine color for PFR display.

        Args:
            stats: Player statistics.

        Returns:
            Color category based on PFR thresholds.
        """
        if stats.pfr > self._config.pfr_loose_threshold:
            return StatColor.RED
        if stats.pfr < self._config.pfr_tight_threshold:
            return StatColor.RED
        if stats.pfr > 20.0:
            return StatColor.YELLOW
        return StatColor.GREEN

    def get_overall_color(self, stats: PlayerStats) -> StatColor:
        """Determine overall color for the stat display widget.

        Uses VPIP as the primary indicator for overall player type.

        Args:
            stats: Player statistics.

        Returns:
            Overall color category.
        """
        return self.get_vpip_color(stats)

    @staticmethod
    def _format_pct(value: float, decimals: int = 0) -> str:
        """Format a percentage value.

        Args:
            value: Percentage value (0-100).
            decimals: Number of decimal places.

        Returns:
            Formatted string.
        """
        if decimals == 0:
            return str(int(round(value)))
        return f"{value:.{decimals}f}"

    @staticmethod
    def _format_af(value: float) -> str:
        """Format an aggression factor value.

        Args:
            value: Aggression factor.

        Returns:
            Formatted string with 1 decimal place.
        """
        return f"{value:.1f}"


@dataclass
class SeatStats:
    """Stats display state for a single seat.

    Attributes:
        seat: The seat position on the table.
        stats: Current player stats (None if seat is empty).
        formatted_text: Pre-formatted display text.
        color: Display color based on stat thresholds.
    """

    seat: SeatPosition = field(default_factory=SeatPosition)
    stats: PlayerStats | None = None
    formatted_text: str = ""
    color: StatColor = StatColor.WHITE


class HUDStatsWidget:
    """Manages per-player stat display widgets for all seats.

    Creates and updates stat displays near each player's seat position
    on the poker table.

    Args:
        table_size: Number of seats at the table (2-10).
        window_width: Poker window width in points.
        window_height: Poker window height in points.
        stats_config: Stats threshold configuration.
        compact: Whether to use compact display format.
        font_size: Font size for stat text.
    """

    def __init__(
        self,
        table_size: int = 9,
        window_width: float = 800.0,
        window_height: float = 600.0,
        stats_config: StatsConfig | None = None,
        compact: bool = True,
        font_size: float = 12.0,
    ) -> None:
        self._table_size = table_size
        self._window_width = window_width
        self._window_height = window_height
        self._compact = compact
        self._font_size = font_size
        self._formatter = StatsFormatter(stats_config)
        self._seats: dict[int, SeatStats] = {}

        # Initialize seat positions
        positions = get_seat_positions(
            table_size, window_width, window_height
        )
        for pos in positions:
            self._seats[pos.seat_number] = SeatStats(seat=pos)

    @property
    def table_size(self) -> int:
        """Return the number of seats at the table."""
        return self._table_size

    @property
    def seats(self) -> dict[int, SeatStats]:
        """Return the current seat stats."""
        return self._seats

    @property
    def formatter(self) -> StatsFormatter:
        """Return the stats formatter."""
        return self._formatter

    @property
    def compact(self) -> bool:
        """Return whether compact mode is enabled."""
        return self._compact

    @compact.setter
    def compact(self, value: bool) -> None:
        """Set compact mode and reformat all stats."""
        self._compact = value
        self._reformat_all()

    def update_stats(self, seat: int, stats: PlayerStats) -> None:
        """Update the stats display for a specific seat.

        Args:
            seat: Seat number (1-based).
            stats: New player statistics.

        Raises:
            ValueError: If seat number is out of range.
        """
        if seat not in self._seats:
            raise ValueError(
                f"Seat {seat} not found. Valid seats: "
                f"1-{self._table_size}"
            )

        seat_stats = self._seats[seat]
        seat_stats.stats = stats

        if self._compact:
            seat_stats.formatted_text = self._formatter.format_compact(stats)
        else:
            seat_stats.formatted_text = self._formatter.format_detailed(stats)

        seat_stats.color = self._formatter.get_overall_color(stats)

        logger.debug(
            "Updated seat %d stats: %s (%s)",
            seat,
            seat_stats.formatted_text,
            seat_stats.color.name,
        )

    def clear_seat(self, seat: int) -> None:
        """Clear stats for a specific seat (player left).

        Args:
            seat: Seat number (1-based).

        Raises:
            ValueError: If seat number is out of range.
        """
        if seat not in self._seats:
            raise ValueError(
                f"Seat {seat} not found. Valid seats: "
                f"1-{self._table_size}"
            )
        self._seats[seat].stats = None
        self._seats[seat].formatted_text = ""
        self._seats[seat].color = StatColor.WHITE

    def clear_all(self) -> None:
        """Clear stats for all seats."""
        for seat_num in self._seats:
            self._seats[seat_num].stats = None
            self._seats[seat_num].formatted_text = ""
            self._seats[seat_num].color = StatColor.WHITE

    def get_display_text(self, seat: int) -> str:
        """Get the formatted display text for a seat.

        Args:
            seat: Seat number (1-based).

        Returns:
            Formatted stat string, or empty string if no stats.
        """
        if seat in self._seats:
            return self._seats[seat].formatted_text
        return ""

    def get_display_color(self, seat: int) -> StatColor:
        """Get the display color for a seat.

        Args:
            seat: Seat number (1-based).

        Returns:
            Color category for the seat's stats.
        """
        if seat in self._seats:
            return self._seats[seat].color
        return StatColor.WHITE

    def get_all_display_data(self) -> list[tuple[SeatPosition, str, StatColor]]:
        """Get display data for all occupied seats.

        Returns:
            List of (position, text, color) tuples for seats with stats.
        """
        result = []
        for seat_stats in self._seats.values():
            if seat_stats.stats is not None:
                result.append(
                    (
                        seat_stats.seat,
                        seat_stats.formatted_text,
                        seat_stats.color,
                    )
                )
        return result

    def _reformat_all(self) -> None:
        """Reformat all seat stats with current display mode."""
        for seat_stats in self._seats.values():
            if seat_stats.stats is not None:
                if self._compact:
                    seat_stats.formatted_text = (
                        self._formatter.format_compact(seat_stats.stats)
                    )
                else:
                    seat_stats.formatted_text = (
                        self._formatter.format_detailed(seat_stats.stats)
                    )
