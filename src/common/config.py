"""Application configuration dataclasses.

Central configuration for all subsystems, loadable/saveable to JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for the overlay window appearance.

    Attributes:
        opacity: Window opacity 0.0 (invisible) to 1.0 (opaque).
        font_size: Base font size in points.
        compact_mode: Use compact stat display format.
        show_stats_panel: Whether to show the stats panel.
        show_solver_panel: Whether to show the solver panel.
        show_settings_panel: Whether to show the settings panel.
    """

    opacity: float = 0.8
    font_size: float = 14.0
    compact_mode: bool = True
    show_stats_panel: bool = True
    show_solver_panel: bool = True
    show_settings_panel: bool = False


@dataclass
class StatsConfig:
    """Configuration for stat display thresholds and colors.

    Attributes:
        vpip_loose_threshold: VPIP above this is considered loose.
        vpip_tight_threshold: VPIP below this is considered tight.
        pfr_loose_threshold: PFR above this is considered loose.
        pfr_tight_threshold: PFR below this is considered tight.
        three_bet_high_threshold: 3-Bet above this is considered high.
        af_high_threshold: Aggression factor above this is high.
    """

    vpip_loose_threshold: float = 30.0
    vpip_tight_threshold: float = 15.0
    pfr_loose_threshold: float = 25.0
    pfr_tight_threshold: float = 12.0
    three_bet_high_threshold: float = 10.0
    af_high_threshold: float = 3.0


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        overlay: Overlay window settings.
        stats: Stat display threshold settings.
        poker_client: Name of the poker client (e.g. "PokerStars", "888poker").
        db_path: Path to the SQLite database file.
        table_size: Number of seats at the table (2-10).
    """

    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    poker_client: str = "PokerStars"
    db_path: str = "data/poker_hud.db"
    table_size: int = 9

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file.

        Args:
            path: File path to write the JSON configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Configuration saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        """Load configuration from a JSON file.

        Args:
            path: File path to read the JSON configuration from.

        Returns:
            Loaded AppConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        overlay_data = data.get("overlay", {})
        stats_data = data.get("stats", {})

        return cls(
            overlay=OverlayConfig(**overlay_data),
            stats=StatsConfig(**stats_data),
            poker_client=data.get("poker_client", "PokerStars"),
            db_path=data.get("db_path", "data/poker_hud.db"),
            table_size=data.get("table_size", 9),
        )
