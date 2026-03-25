
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poker HUD (Heads-Up Display) — a macOS desktop application that captures a live poker table, recognizes cards and actions via ML/CV, tracks opponent statistics, provides GTO (Game Theory Optimal) solver recommendations, and renders an overlay HUD on top of the poker client window.

## Planned Architecture

The project is organized into six subsystems under `src/`:

- **capture/** — Screen capture (macOS ScreenCaptureKit/PyObjC), poker window auto-detection, smart polling with frame change detection, and a capture pipeline coordinator
- **detection/** — Card recognition CNN model, card region localization on the table, and end-to-end card recognition pipeline. Training data and trained models live in `data/` and `models/`
- **engine/** — Game state data model, hand phase/street tracking, player action detection (screen analysis), player identification (OCR on screen names), game state coordinator
- **stats/** — SQLite-backed opponent statistics: VPIP, PFR, 3-Bet%, Fold-to-3-Bet, C-Bet%, Aggression Factor, WTSD%. Uses repository pattern for DB access. Real-time stats aggregation service
- **solver/** — Preflop GTO range lookup tables, hand equity calculator, postflop decision engine, opponent range estimation, real-time GTO recommendations
- **overlay/** — Transparent macOS overlay window, HUD stat display widgets, GTO solver panel, settings/configuration UI

## Build & Development Commands

The project targets Python 3.11+ and uses `pyproject.toml` for dependency management.

```bash
make test        # Run all tests (pytest)
make lint        # Run linter (ruff)
make format      # Auto-format code (black)
```

Tests are organized as `tests/unit/`, `tests/integration/`, and `tests/fixtures/`.

## Key Technical Details

- macOS-only: relies on ScreenCaptureKit (via PyObjC) for screen capture, requires Screen Recording permission
- Frames are numpy arrays in BGR format (OpenCV-compatible)
- Card detection uses a CNN trained on a custom dataset
- Stats stored in SQLite with repository pattern
- Overlay uses a transparent always-on-top macOS window
- Target latency: <500ms end-to-end (capture → HUD update), <100ms per frame capture
- Project tracking is on Monday.com (board ID 5027412713), tickets created via `create_tickets.py`
