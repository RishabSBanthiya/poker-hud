"""Application configuration management.

Loads configuration from YAML/JSON files and environment variables,
providing typed access to settings for all subsystems.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for the capture subsystem.

    Attributes:
        poll_interval_ms: Milliseconds between frame captures.
        frame_change_threshold: Minimum pixel difference to consider a frame changed.
        window_title_pattern: Regex pattern to match the poker client window title.
    """

    poll_interval_ms: int = 200
    frame_change_threshold: float = 0.01
    window_title_pattern: str = ".*[Pp]oker.*"


@dataclass
class DetectionConfig:
    """Configuration for the detection subsystem.

    Attributes:
        confidence_threshold: Minimum confidence for card detection.
        template_dir: Path to card template images.
        ocr_enabled: Whether to run OCR for player names.
    """

    confidence_threshold: float = 0.8
    template_dir: str = "data/templates"
    ocr_enabled: bool = True


@dataclass
class StatsConfig:
    """Configuration for the stats subsystem.

    Attributes:
        db_path: Path to the SQLite database file.
        min_hands_for_stats: Minimum hands before showing stats for a player.
    """

    db_path: str = "data/poker_hud.db"
    min_hands_for_stats: int = 5


@dataclass
class SolverConfig:
    """Configuration for the solver subsystem.

    Attributes:
        preflop_ranges_path: Path to preflop range lookup tables.
        equity_simulations: Number of Monte Carlo simulations for equity.
    """

    preflop_ranges_path: str = "data/ranges"
    equity_simulations: int = 10000


@dataclass
class OverlayAppConfig:
    """Configuration for the overlay subsystem.

    Attributes:
        enabled: Whether to show the overlay window.
        font_size: Font size for HUD text.
        opacity: Background opacity (0.0-1.0).
    """

    enabled: bool = True
    font_size: float = 18.0
    opacity: float = 0.6


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        debug: Enable debug mode with verbose logging.
        capture: Capture subsystem configuration.
        detection: Detection subsystem configuration.
        stats: Stats subsystem configuration.
        solver: Solver subsystem configuration.
        overlay: Overlay subsystem configuration.
    """

    debug: bool = False
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    overlay: OverlayAppConfig = field(default_factory=OverlayAppConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> AppConfig:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            An AppConfig instance populated from the file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Create an AppConfig from a dictionary.

        Args:
            data: Dictionary of configuration values.

        Returns:
            An AppConfig instance.
        """
        config = cls()
        config.debug = data.get("debug", config.debug)

        if "capture" in data:
            cap = data["capture"]
            config.capture = CaptureConfig(
                poll_interval_ms=cap.get(
                    "poll_interval_ms", config.capture.poll_interval_ms
                ),
                frame_change_threshold=cap.get(
                    "frame_change_threshold",
                    config.capture.frame_change_threshold,
                ),
                window_title_pattern=cap.get(
                    "window_title_pattern",
                    config.capture.window_title_pattern,
                ),
            )

        if "detection" in data:
            det = data["detection"]
            config.detection = DetectionConfig(
                confidence_threshold=det.get(
                    "confidence_threshold",
                    config.detection.confidence_threshold,
                ),
                template_dir=det.get(
                    "template_dir", config.detection.template_dir
                ),
                ocr_enabled=det.get(
                    "ocr_enabled", config.detection.ocr_enabled
                ),
            )

        if "stats" in data:
            st = data["stats"]
            config.stats = StatsConfig(
                db_path=st.get("db_path", config.stats.db_path),
                min_hands_for_stats=st.get(
                    "min_hands_for_stats",
                    config.stats.min_hands_for_stats,
                ),
            )

        if "solver" in data:
            sol = data["solver"]
            config.solver = SolverConfig(
                preflop_ranges_path=sol.get(
                    "preflop_ranges_path",
                    config.solver.preflop_ranges_path,
                ),
                equity_simulations=sol.get(
                    "equity_simulations",
                    config.solver.equity_simulations,
                ),
            )

        if "overlay" in data:
            ov = data["overlay"]
            config.overlay = OverlayAppConfig(
                enabled=ov.get("enabled", config.overlay.enabled),
                font_size=ov.get("font_size", config.overlay.font_size),
                opacity=ov.get("opacity", config.overlay.opacity),
            )

        return config
