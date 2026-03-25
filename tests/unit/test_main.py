"""Unit tests for the main entry point."""

from __future__ import annotations

from main import parse_args


class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_default_args(self) -> None:
        args = parse_args([])
        assert args.config is None
        assert args.debug is False
        assert args.no_overlay is False

    def test_config_arg(self) -> None:
        args = parse_args(["--config", "my_config.json"])
        assert args.config == "my_config.json"

    def test_debug_flag(self) -> None:
        args = parse_args(["--debug"])
        assert args.debug is True

    def test_no_overlay_flag(self) -> None:
        args = parse_args(["--no-overlay"])
        assert args.no_overlay is True

    def test_all_args(self) -> None:
        args = parse_args(["--config", "c.json", "--debug", "--no-overlay"])
        assert args.config == "c.json"
        assert args.debug is True
        assert args.no_overlay is True
