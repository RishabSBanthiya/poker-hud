"""Configuration management system for poker-hud.

Provides typed, dataclass-based configuration with sensible defaults. Supports
loading overrides from YAML files, TOML files, and environment variables.

Environment variables use the prefix ``POKERHUD_`` and double underscores for
nesting. For example, ``POKERHUD_CAPTURE__FPS=30`` sets ``capture.fps``.

Usage:
    from src.common.config import load_config

    config = load_config()                       # defaults only
    config = load_config("settings.yaml")        # override from file
    config = load_config(env_prefix="POKERHUD")  # merge env vars
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureConfig:
    """Settings for the screen capture subsystem."""

    fps: int = 10
    resolution_width: int = 1920
    resolution_height: int = 1080
    polling_interval_ms: int = 100
    change_threshold: float = 0.01


@dataclass(frozen=True)
class DetectionConfig:
    """Settings for the card detection subsystem."""

    confidence_threshold: float = 0.85
    template_path: str = "data/templates"
    model_path: str = "models/card_detector.pt"
    input_size: int = 224
    batch_size: int = 1


@dataclass(frozen=True)
class OverlayConfig:
    """Settings for the HUD overlay."""

    position_x: int = 0
    position_y: int = 0
    opacity: float = 0.9
    font_size: int = 14
    background_color: str = "#1a1a2e"
    text_color: str = "#e0e0e0"
    accent_color: str = "#00d4aa"


@dataclass(frozen=True)
class StatsConfig:
    """Settings for the statistics / database subsystem."""

    db_path: str = "data/poker_hud.db"
    max_connections: int = 5
    wal_mode: bool = True

    # HUD display thresholds for color coding
    vpip_loose_threshold: float = 40.0
    vpip_tight_threshold: float = 12.0
    pfr_loose_threshold: float = 30.0
    pfr_tight_threshold: float = 8.0


@dataclass(frozen=True)
class SolverConfig:
    """Settings for the GTO solver subsystem."""

    preflop_table_path: str = "data/preflop_ranges"
    equity_iterations: int = 10000
    cache_size: int = 1024


@dataclass(frozen=True)
class GeneralConfig:
    """Top-level application settings."""

    log_level: str = "INFO"
    debug: bool = False
    json_logging: bool = False
    log_file: str | None = None


@dataclass(frozen=True)
class AppConfig:
    """Root configuration aggregating all subsystem configs.

    Attributes:
        general: Top-level application settings.
        capture: Screen capture subsystem settings.
        detection: Card detection subsystem settings.
        overlay: HUD overlay settings.
        stats: Statistics / database subsystem settings.
        solver: GTO solver subsystem settings.
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *base*, returning a new dict."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_value(value: str, target_type: type) -> Any:
    """Coerce a string environment variable to the field's declared type."""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _load_env_overrides(prefix: str) -> dict[str, dict[str, str]]:
    """Collect environment variables matching *prefix* into a nested dict.

    ``POKERHUD_CAPTURE__FPS=30`` becomes ``{"capture": {"fps": "30"}}``.
    """
    result: dict[str, dict[str, str]] = {}
    prefix_upper = prefix.upper() + "_"
    for key, value in os.environ.items():
        if not key.startswith(prefix_upper):
            continue
        remainder = key[len(prefix_upper):].lower()
        parts = remainder.split("__", maxsplit=1)
        if len(parts) == 2:
            section, field_name = parts
            result.setdefault(section, {})[field_name] = value
        # Single-segment keys (e.g. POKERHUD_DEBUG) map to "general"
        elif len(parts) == 1:
            result.setdefault("general", {})[parts[0]] = value
    return result


def _load_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML or TOML configuration file into a dict.

    Args:
        path: Filesystem path. Extension determines the parser used.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            ) from exc
        with open(path) as fh:
            return yaml.safe_load(fh) or {}

    if suffix == ".toml":
        # Python 3.11+ has tomllib in the standard library
        import tomllib

        with open(path, "rb") as fh:
            return tomllib.load(fh)

    raise ValueError(f"Unsupported config file format: {suffix}")


def _dict_to_config(data: dict[str, Any]) -> AppConfig:
    """Build an ``AppConfig`` from a nested dict, ignoring unknown keys."""
    section_map: dict[str, type] = {
        "general": GeneralConfig,
        "capture": CaptureConfig,
        "detection": DetectionConfig,
        "overlay": OverlayConfig,
        "stats": StatsConfig,
        "solver": SolverConfig,
    }
    kwargs: dict[str, Any] = {}
    for section_name, section_cls in section_map.items():
        section_data = data.get(section_name, {})
        if not isinstance(section_data, dict):
            continue
        valid_fields = {f.name for f in fields(section_cls)}
        # Coerce env-var string values to the correct types
        coerced: dict[str, Any] = {}
        field_types = {f.name: f.type for f in fields(section_cls)}
        for k, v in section_data.items():
            if k not in valid_fields:
                continue
            if isinstance(v, str):
                target = field_types[k]
                # Resolve string type annotations
                type_map = {
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "str": str,
                }
                if isinstance(target, str):
                    resolved = type_map.get(target, target)
                else:
                    resolved = target
                coerced[k] = _coerce_value(v, resolved)
            else:
                coerced[k] = v
        kwargs[section_name] = section_cls(**coerced)
    return AppConfig(**kwargs)


def load_config(
    config_path: str | Path | None = None,
    *,
    env_prefix: str = "POKERHUD",
) -> AppConfig:
    """Load application configuration with layered overrides.

    Resolution order (later wins):
      1. Dataclass defaults
      2. Config file (YAML / TOML)
      3. Environment variables

    Args:
        config_path: Optional path to a YAML or TOML config file.
        env_prefix: Prefix for environment variable overrides.

    Returns:
        Fully resolved ``AppConfig`` instance.
    """
    base = asdict(AppConfig())

    if config_path is not None:
        file_data = _load_file(config_path)
        base = _deep_merge(base, file_data)

    env_data = _load_env_overrides(env_prefix)
    if env_data:
        base = _deep_merge(base, env_data)

    return _dict_to_config(base)
