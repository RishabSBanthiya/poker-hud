"""HUD settings and configuration UI panel.

Provides settings management for the overlay, including visibility
toggles, opacity control, poker client selection, stat display
preferences, and persistent configuration save/load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.common.config import AppConfig, StatsConfig

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
class SettingsState:
    """Current state of the settings panel UI.

    Attributes:
        visible: Whether the settings panel is shown.
        config: The current application configuration.
        dirty: Whether unsaved changes exist.
        config_path: Path to the config file.
    """

    visible: bool = False
    config: AppConfig = field(default_factory=AppConfig)
    dirty: bool = False
    config_path: str = DEFAULT_CONFIG_PATH


class SettingsPanel:
    """Settings panel for configuring the HUD overlay.

    Manages application configuration including overlay visibility,
    opacity, poker client selection, stat display preferences,
    and database path. Supports save/load to a JSON config file.

    Args:
        config: Initial application configuration.
        config_path: Path to save/load configuration file.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        self._state = SettingsState(
            config=config or AppConfig(),
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
        """Get the current application configuration.

        Returns:
            Current AppConfig instance.
        """
        return self._state.config

    def apply_config(self, config: AppConfig) -> None:
        """Apply a new configuration.

        Args:
            config: New application configuration to apply.
        """
        self._state.config = config
        self._state.dirty = True
        logger.info("Configuration applied (unsaved)")

    # --- Overlay settings ---

    def set_overlay_visible(self, visible: bool) -> None:
        """Toggle overlay visibility in configuration.

        Args:
            visible: Whether the overlay should be visible.
        """
        # Stats and solver panels follow overlay visibility
        self._state.config.overlay.show_stats_panel = visible
        self._state.config.overlay.show_solver_panel = visible
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
        self._state.config.overlay.opacity = opacity
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
        self._state.config.overlay.font_size = font_size
        self._state.dirty = True

    def set_compact_mode(self, compact: bool) -> None:
        """Set compact stat display mode.

        Args:
            compact: True for compact mode, False for detailed.
        """
        self._state.config.overlay.compact_mode = compact
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
        self._state.config.poker_client = client
        self._state.dirty = True

    def get_poker_client(self) -> str:
        """Get the currently selected poker client.

        Returns:
            Name of the selected poker client.
        """
        return self._state.config.poker_client

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
        self._state.config.db_path = path
        self._state.dirty = True

    def get_db_path(self) -> str:
        """Get the current database path.

        Returns:
            Path to the SQLite database file.
        """
        return self._state.config.db_path

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
        self._state.config.table_size = size
        self._state.dirty = True

    # --- Stats threshold settings ---

    def set_stats_config(self, stats_config: StatsConfig) -> None:
        """Set stat display threshold configuration.

        Args:
            stats_config: New stats threshold configuration.
        """
        self._state.config.stats = stats_config
        self._state.dirty = True

    # --- Persistence ---

    def save_config(self, path: str | None = None) -> None:
        """Save current configuration to a JSON file.

        Args:
            path: Optional override path. Uses default if not provided.
        """
        save_path = path or self._state.config_path
        self._state.config.save(save_path)
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
        config = AppConfig.load(load_path)
        self._state.config = config
        self._state.dirty = False
        logger.info("Configuration loaded from %s", load_path)
        return config

    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self._state.config = AppConfig()
        self._state.dirty = True
        logger.info("Configuration reset to defaults")

    def get_settings_summary(self) -> str:
        """Get a human-readable summary of current settings.

        Returns:
            Multi-line settings summary string.
        """
        cfg = self._state.config
        lines = [
            f"Poker Client: {cfg.poker_client}",
            f"Table Size: {cfg.table_size}",
            f"Opacity: {cfg.overlay.opacity:.0%}",
            f"Font Size: {cfg.overlay.font_size:.0f}pt",
            f"Compact Mode: {'On' if cfg.overlay.compact_mode else 'Off'}",
            f"Stats Panel: {'On' if cfg.overlay.show_stats_panel else 'Off'}",
            f"Solver Panel: {'On' if cfg.overlay.show_solver_panel else 'Off'}",
            f"DB Path: {cfg.db_path}",
        ]
        return "\n".join(lines)
