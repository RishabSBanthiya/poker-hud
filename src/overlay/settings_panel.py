"""HUD settings and configuration UI panel.

Provides settings management for the overlay, including visibility
toggles, opacity control, poker client selection, stat display
preferences, and persistent configuration save/load.

Internally stores mutable settings state that can be converted to/from
the frozen ``AppConfig`` dataclass via ``get_config()`` and JSON
serialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.config import (
    AppConfig,
    OverlayConfig,
    StatsConfig,
)

logger = logging.getLogger(__name__)

# Supported poker clients
SUPPORTED_CLIENTS: list[str] = [
    "PokerStars",
    "888poker",
    "PartyPoker",
    "GGPoker",
    "WPN",
    "Winamax",
]

# Default config file path
DEFAULT_CONFIG_PATH = "config/settings.json"


@dataclass
class MutableSettings:
    """Mutable internal settings state.

    These fields mirror what the UI exposes.  Fields that don't exist on
    ``AppConfig`` (poker_client, table_size, compact_mode,
    show_stats_panel, show_solver_panel) are stored here but serialized
    alongside the config sections in JSON.

    Attributes:
        opacity: Overlay opacity (0.0-1.0).
        font_size: Overlay font size in points.
        position_x: Overlay X position.
        position_y: Overlay Y position.
        background_color: Overlay background color hex.
        text_color: Overlay text color hex.
        accent_color: Overlay accent color hex.
        db_path: Path to the SQLite database.
        poker_client: Selected poker client name.
        table_size: Number of seats at the table.
        compact_mode: Whether to use compact stat display.
        show_stats_panel: Whether to show the stats panel.
        show_solver_panel: Whether to show the solver panel.
        stats_config: Stats threshold configuration.
    """

    opacity: float = 0.9
    font_size: int = 14
    position_x: int = 0
    position_y: int = 0
    background_color: str = "#1a1a2e"
    text_color: str = "#e0e0e0"
    accent_color: str = "#00d4aa"
    db_path: str = "data/poker_hud.db"
    poker_client: str = "PokerStars"
    table_size: int = 9
    compact_mode: bool = False
    show_stats_panel: bool = True
    show_solver_panel: bool = True
    stats_config: StatsConfig = field(default_factory=StatsConfig)


def _settings_from_config(config: AppConfig) -> MutableSettings:
    """Extract mutable settings from a frozen AppConfig.

    Args:
        config: Frozen application configuration.

    Returns:
        A new MutableSettings populated from the config.
    """
    return MutableSettings(
        opacity=config.overlay.opacity,
        font_size=config.overlay.font_size,
        position_x=config.overlay.position_x,
        position_y=config.overlay.position_y,
        background_color=config.overlay.background_color,
        text_color=config.overlay.text_color,
        accent_color=config.overlay.accent_color,
        db_path=config.stats.db_path,
        stats_config=config.stats,
    )


def _settings_to_config(settings: MutableSettings) -> AppConfig:
    """Build an AppConfig from mutable settings.

    Args:
        settings: Current mutable settings.

    Returns:
        A frozen AppConfig instance.
    """
    return AppConfig(
        overlay=OverlayConfig(
            opacity=settings.opacity,
            font_size=settings.font_size,
            position_x=settings.position_x,
            position_y=settings.position_y,
            background_color=settings.background_color,
            text_color=settings.text_color,
            accent_color=settings.accent_color,
        ),
        stats=settings.stats_config,
    )


@dataclass
class SettingsState:
    """Current state of the settings panel UI.

    Attributes:
        visible: Whether the settings panel is shown.
        settings: The mutable internal settings.
        dirty: Whether unsaved changes exist.
        config_path: Path to the config file.
    """

    visible: bool = False
    settings: MutableSettings = field(default_factory=MutableSettings)
    dirty: bool = False
    config_path: str = DEFAULT_CONFIG_PATH


class SettingsPanel:
    """Settings panel for configuring the HUD overlay.

    Manages application configuration including overlay visibility,
    opacity, poker client selection, stat display preferences,
    and database path. Supports save/load to a JSON config file.

    Internally uses ``MutableSettings`` so values can be changed
    freely, then reconstructs a frozen ``AppConfig`` on demand via
    ``get_config()``.

    Args:
        config: Initial application configuration.
        config_path: Path to save/load configuration file.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        initial_config = config or AppConfig()
        settings = _settings_from_config(initial_config)
        self._state = SettingsState(
            settings=settings,
            config_path=config_path,
        )

    @property
    def state(self) -> SettingsState:
        """Return the current settings state."""
        return self._state

    @property
    def visible(self) -> bool:
        """Return whether the settings panel is visible."""
        return self._state.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set settings panel visibility."""
        self._state.visible = value

    @property
    def is_dirty(self) -> bool:
        """Return whether there are unsaved changes."""
        return self._state.dirty

    def get_config(self) -> AppConfig:
        """Build and return an AppConfig from the current mutable settings.

        Returns:
            Current AppConfig instance.
        """
        return _settings_to_config(self._state.settings)

    def apply_config(self, config: AppConfig) -> None:
        """Apply a new configuration by extracting its values.

        Args:
            config: New application configuration to apply.
        """
        self._state.settings = _settings_from_config(config)
        self._state.dirty = True
        logger.info("Configuration applied (unsaved)")

    # --- Overlay settings ---

    def set_overlay_visible(self, visible: bool) -> None:
        """Toggle overlay sub-panel visibility.

        Args:
            visible: Whether the overlay panels should be visible.
        """
        self._state.settings.show_stats_panel = visible
        self._state.settings.show_solver_panel = visible
        self._state.dirty = True

    def set_opacity(self, opacity: float) -> None:
        """Set overlay opacity.

        Args:
            opacity: Opacity value from 0.0 to 1.0.

        Raises:
            ValueError: If opacity is outside [0.0, 1.0].
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                f"Opacity must be between 0.0 and 1.0, got {opacity}"
            )
        self._state.settings.opacity = opacity
        self._state.dirty = True

    def set_font_size(self, font_size: float) -> None:
        """Set overlay font size.

        Args:
            font_size: Font size in points (must be positive).

        Raises:
            ValueError: If font_size is not positive.
        """
        if font_size <= 0:
            raise ValueError(f"Font size must be positive, got {font_size}")
        self._state.settings.font_size = int(font_size)
        self._state.dirty = True

    def set_compact_mode(self, compact: bool) -> None:
        """Set compact stat display mode.

        Args:
            compact: True for compact mode, False for detailed.
        """
        self._state.settings.compact_mode = compact
        self._state.dirty = True

    # --- Poker client settings ---

    def set_poker_client(self, client: str) -> None:
        """Set the poker client.

        Args:
            client: Name of the poker client.

        Raises:
            ValueError: If client is not in the supported list.
        """
        if client not in SUPPORTED_CLIENTS:
            raise ValueError(
                f"Unsupported client '{client}'. "
                f"Supported: {SUPPORTED_CLIENTS}"
            )
        self._state.settings.poker_client = client
        self._state.dirty = True

    def get_poker_client(self) -> str:
        """Get the currently selected poker client.

        Returns:
            Name of the selected poker client.
        """
        return self._state.settings.poker_client

    # --- Database settings ---

    def set_db_path(self, path: str) -> None:
        """Set the database file path.

        Args:
            path: Path to the SQLite database file.

        Raises:
            ValueError: If path is empty.
        """
        if not path.strip():
            raise ValueError("Database path cannot be empty")
        self._state.settings.db_path = path
        self._state.dirty = True

    def get_db_path(self) -> str:
        """Get the current database path.

        Returns:
            Path to the SQLite database file.
        """
        return self._state.settings.db_path

    # --- Table settings ---

    def set_table_size(self, size: int) -> None:
        """Set the number of seats at the table.

        Args:
            size: Number of seats (2-10).

        Raises:
            ValueError: If size is outside [2, 10].
        """
        if not 2 <= size <= 10:
            raise ValueError(f"Table size must be 2-10, got {size}")
        self._state.settings.table_size = size
        self._state.dirty = True

    # --- Stats threshold settings ---

    def set_stats_config(self, stats_config: StatsConfig) -> None:
        """Set stat display threshold configuration.

        Args:
            stats_config: New stats threshold configuration.
        """
        self._state.settings.stats_config = stats_config
        self._state.dirty = True

    # --- Persistence ---

    def _to_dict(self) -> dict[str, Any]:
        """Serialize the mutable settings to a dict for JSON persistence.

        Returns:
            Nested dict suitable for JSON serialization.
        """
        s = self._state.settings
        return {
            "overlay": {
                "opacity": s.opacity,
                "font_size": s.font_size,
                "position_x": s.position_x,
                "position_y": s.position_y,
                "background_color": s.background_color,
                "text_color": s.text_color,
                "accent_color": s.accent_color,
            },
            "stats": {
                "db_path": s.stats_config.db_path,
                "max_connections": s.stats_config.max_connections,
                "wal_mode": s.stats_config.wal_mode,
                "vpip_loose_threshold": s.stats_config.vpip_loose_threshold,
                "vpip_tight_threshold": s.stats_config.vpip_tight_threshold,
                "pfr_loose_threshold": s.stats_config.pfr_loose_threshold,
                "pfr_tight_threshold": s.stats_config.pfr_tight_threshold,
            },
            "poker_client": s.poker_client,
            "table_size": s.table_size,
            "compact_mode": s.compact_mode,
            "show_stats_panel": s.show_stats_panel,
            "show_solver_panel": s.show_solver_panel,
            "db_path": s.db_path,
        }

    def _from_dict(self, data: dict[str, Any]) -> None:
        """Restore mutable settings from a dict loaded from JSON.

        Args:
            data: Dict previously produced by ``_to_dict``.
        """
        s = self._state.settings
        overlay = data.get("overlay", {})
        s.opacity = overlay.get("opacity", s.opacity)
        s.font_size = overlay.get("font_size", s.font_size)
        s.position_x = overlay.get("position_x", s.position_x)
        s.position_y = overlay.get("position_y", s.position_y)
        s.background_color = overlay.get("background_color", s.background_color)
        s.text_color = overlay.get("text_color", s.text_color)
        s.accent_color = overlay.get("accent_color", s.accent_color)

        stats_data = data.get("stats", {})
        if stats_data:
            s.stats_config = StatsConfig(
                db_path=stats_data.get("db_path", s.stats_config.db_path),
                max_connections=stats_data.get(
                    "max_connections", s.stats_config.max_connections
                ),
                wal_mode=stats_data.get("wal_mode", s.stats_config.wal_mode),
                vpip_loose_threshold=stats_data.get(
                    "vpip_loose_threshold",
                    s.stats_config.vpip_loose_threshold,
                ),
                vpip_tight_threshold=stats_data.get(
                    "vpip_tight_threshold",
                    s.stats_config.vpip_tight_threshold,
                ),
                pfr_loose_threshold=stats_data.get(
                    "pfr_loose_threshold",
                    s.stats_config.pfr_loose_threshold,
                ),
                pfr_tight_threshold=stats_data.get(
                    "pfr_tight_threshold",
                    s.stats_config.pfr_tight_threshold,
                ),
            )

        s.poker_client = data.get("poker_client", s.poker_client)
        s.table_size = data.get("table_size", s.table_size)
        s.compact_mode = data.get("compact_mode", s.compact_mode)
        s.show_stats_panel = data.get("show_stats_panel", s.show_stats_panel)
        s.show_solver_panel = data.get("show_solver_panel", s.show_solver_panel)
        s.db_path = data.get("db_path", s.db_path)

    def save_config(self, path: str | None = None) -> None:
        """Save current configuration to a JSON file.

        Args:
            path: Optional override path. Uses default if not provided.
        """
        save_path = path or self._state.config_path
        data = self._to_dict()
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        save_file.write_text(json.dumps(data, indent=2))
        self._state.dirty = False
        logger.info("Configuration saved to %s", save_path)

    def load_config(self, path: str | None = None) -> AppConfig:
        """Load configuration from a JSON file.

        Args:
            path: Optional override path. Uses default if not provided.

        Returns:
            The loaded AppConfig.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file is invalid.
        """
        load_path = path or self._state.config_path
        load_file = Path(load_path)
        if not load_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {load_path}"
            )
        data = json.loads(load_file.read_text())
        self._from_dict(data)
        self._state.dirty = False
        logger.info("Configuration loaded from %s", load_path)
        return self.get_config()

    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self._state.settings = MutableSettings()
        self._state.dirty = True
        logger.info("Configuration reset to defaults")

    def get_settings_summary(self) -> str:
        """Get a human-readable summary of current settings.

        Returns:
            Multi-line settings summary string.
        """
        s = self._state.settings
        lines = [
            f"Poker Client: {s.poker_client}",
            f"Table Size: {s.table_size}",
            f"Opacity: {s.opacity:.0%}",
            f"Font Size: {s.font_size}pt",
            f"Compact Mode: {'On' if s.compact_mode else 'Off'}",
            f"Stats Panel: {'On' if s.show_stats_panel else 'Off'}",
            f"Solver Panel: {'On' if s.show_solver_panel else 'Off'}",
            f"DB Path: {s.db_path}",
        ]
        return "\n".join(lines)
