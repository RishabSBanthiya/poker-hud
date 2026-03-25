"""Main application class for the Poker HUD.

Orchestrates initialization, wiring, and lifecycle management
of all subsystems: capture, detection, engine, stats, solver, overlay.
"""

from __future__ import annotations

import logging
import signal
import threading
from enum import Enum
from typing import Callable, Optional

import numpy as np

from src.capture.pipeline import CapturePipeline, PipelineConfig
from src.capture.window_detector import WindowInfo
from src.common.config import AppConfig
from src.detection.card_recognition import CardRecognitionPipeline
from src.detection.ocr_engine import OCREngine
from src.detection.player_identifier import PlayerIdentifier
from src.detection.validation import DetectionResult
from src.engine.coordinator import GameStateCoordinator, StateChangeEvent
from src.engine.game_state import GameState
from src.overlay.hud_stats import StatsFormatter
from src.overlay.overlay_window import (
    OverlayConfig,
    OverlayWindow,
    PanelType,
    WindowInfo as OverlayWindowInfo,
)
from src.solver.advisor_coordinator import (
    ActionRecommendation,
    StrategyAdvisorCoordinator,
    StrategyAdvice,
)
from src.stats.aggregator import StatsAggregator
from src.stats.connection import ConnectionManager

logger = logging.getLogger(__name__)


class AppState(Enum):
    """Application lifecycle states."""

    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class PokerHUDApp:
    """Main application coordinating all Poker HUD subsystems.

    Manages the full lifecycle: initialization, wiring, start/stop,
    pause/resume, and graceful shutdown with signal handling.

    Args:
        config: Application configuration. Uses defaults if None.
        enable_overlay: Whether to create the overlay window.
        debug: Whether to enable debug-level logging.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        enable_overlay: bool = True,
        debug: bool = False,
    ) -> None:
        self._config = config or AppConfig()
        self._enable_overlay = enable_overlay
        self._debug = debug
        self._state = AppState.CREATED
        self._shutdown_event = threading.Event()

        # Subsystem references (initialized in initialize())
        self._connection_manager: Optional[ConnectionManager] = None
        self._capture_pipeline: Optional[CapturePipeline] = None
        self._card_recognition: Optional[CardRecognitionPipeline] = None
        self._ocr_engine: Optional[OCREngine] = None
        self._player_identifier: Optional[PlayerIdentifier] = None
        self._game_state_coordinator: Optional[GameStateCoordinator] = None
        self._stats_aggregator: Optional[StatsAggregator] = None
        self._strategy_advisor: Optional[StrategyAdvisorCoordinator] = None
        self._overlay: Optional[OverlayWindow] = None
        self._stats_formatter: Optional[StatsFormatter] = None

    @property
    def state(self) -> AppState:
        """Current application state."""
        return self._state

    @property
    def config(self) -> AppConfig:
        """Application configuration."""
        return self._config

    @property
    def connection_manager(self) -> Optional[ConnectionManager]:
        """Database connection manager."""
        return self._connection_manager

    @property
    def capture_pipeline(self) -> Optional[CapturePipeline]:
        """Capture pipeline instance."""
        return self._capture_pipeline

    @property
    def game_state_coordinator(self) -> Optional[GameStateCoordinator]:
        """Game state coordinator instance."""
        return self._game_state_coordinator

    @property
    def stats_aggregator(self) -> Optional[StatsAggregator]:
        """Stats aggregator instance."""
        return self._stats_aggregator

    @property
    def strategy_advisor(self) -> Optional[StrategyAdvisorCoordinator]:
        """Strategy advisor instance."""
        return self._strategy_advisor

    @property
    def overlay(self) -> Optional[OverlayWindow]:
        """Overlay window instance."""
        return self._overlay

    def initialize(self) -> None:
        """Initialize all subsystems in dependency order.

        1. Logging
        2. Database / connection manager
        3. Capture pipeline
        4. Detection components (card recognition, OCR, player ID)
        5. Game state engine (coordinator)
        6. Stats aggregator
        7. Strategy advisor
        8. Wire subsystems together via callbacks
        9. Overlay (if enabled)
        """
        if self._state not in (AppState.CREATED, AppState.STOPPED):
            logger.warning(
                "Cannot initialize in state %s", self._state.value
            )
            return

        logger.info("Initializing Poker HUD application...")

        # 1. Logging setup
        self._setup_logging()

        # 2. Database
        self._connection_manager = ConnectionManager(self._config.stats)

        # 3. Capture pipeline
        pipeline_config = PipelineConfig(
            polling_interval=self._config.capture.polling_interval_ms / 1000.0,
            change_threshold=self._config.capture.change_threshold,
        )
        self._capture_pipeline = CapturePipeline(config=pipeline_config)

        # 4. Detection components
        self._card_recognition = CardRecognitionPipeline(
            template_dir=self._config.detection.template_path,
            confidence_threshold=self._config.detection.confidence_threshold,
        )
        self._card_recognition.initialize()

        self._ocr_engine = OCREngine()

        self._player_identifier = PlayerIdentifier()

        # 5. Game state engine
        self._game_state_coordinator = GameStateCoordinator()

        # 6. Stats aggregator
        self._stats_aggregator = StatsAggregator()

        # 7. Strategy advisor
        self._strategy_advisor = StrategyAdvisorCoordinator()

        # 8. Wire subsystems together
        self._wire_subsystems()

        # 9. Overlay (if enabled)
        if self._enable_overlay:
            # Position overlay in the top-right of the screen initially
            screen_w, screen_h = OverlayWindow.screen_size()
            overlay_w = 520.0
            overlay_h = 50.0
            margin = 20.0
            overlay_config = OverlayConfig(
                x=screen_w - overlay_w - margin,
                y=screen_h - overlay_h - margin,
                width=overlay_w,
                height=overlay_h,
                font_size=16.0,
                text_color=(0.0, 1.0, 0.4, 1.0),
                bg_color=(0.1, 0.1, 0.1, 0.75),
            )
            self._overlay = OverlayWindow(
                config=overlay_config,
                text="Poker HUD — Waiting for table...",
            )
            self._stats_formatter = StatsFormatter(self._config.stats)
            self._overlay.create()
            logger.info("Overlay window created")

        self._state = AppState.INITIALIZED
        logger.info("Poker HUD application initialized successfully")

    def _wire_subsystems(self) -> None:
        """Connect subsystems via callbacks to form the data pipeline.

        Data flow:
            Capture -> Detection -> Game State Coordinator -> Stats + Solver -> Overlay
        """
        # Capture -> frame handler (runs detection + feeds coordinator)
        if self._capture_pipeline:
            self._capture_pipeline.register_handler(
                self._on_frame_captured
            )

        # Game State -> Stats + Solver: state change events
        if self._game_state_coordinator:
            if self._stats_aggregator:
                self._game_state_coordinator.on_new_hand(
                    self._on_new_hand_event
                )
            if self._strategy_advisor:
                self._game_state_coordinator.on_state_change(
                    self._on_state_change_event
                )

        # Solver -> Overlay: advice ready callback
        if self._strategy_advisor:
            self._strategy_advisor.on_advice_ready(
                self._on_advice_ready
            )

    def _on_frame_captured(
        self, frame: np.ndarray, window: WindowInfo
    ) -> None:
        """Handle a newly captured frame.

        Runs the detection pipeline on the frame, feeds results into
        the game state coordinator, and repositions the overlay.

        Args:
            frame: BGR numpy array of the captured frame.
            window: Info about the source window.
        """
        logger.debug(
            "Frame captured from '%s' (%dx%d)",
            window.title,
            frame.shape[1],
            frame.shape[0],
        )

        # Keep overlay positioned over the poker window (must run on main thread)
        if self._overlay is not None:
            overlay_win = OverlayWindowInfo(
                x=window.x,
                y=window.y,
                width=window.width,
                height=window.height,
                title=window.title,
                window_id=window.window_id,
            )
            overlay = self._overlay
            title = window.title

            def _reposition() -> None:
                overlay.attach_to_window(overlay_win)
                # Update main text to show we're tracking a table
                if overlay.text.startswith("Poker HUD"):
                    overlay.set_text(f"Tracking: {title}")

            self._run_on_main_thread(_reposition)

        # Run card recognition on the frame
        detection_result = DetectionResult()
        if self._card_recognition is not None:
            card_result = self._card_recognition.process_frame(frame)
            detection_result.community_cards = [
                d.card for d in card_result.community_cards
            ]
            # Hole cards go to the hero seat (seat 0 by default)
            if card_result.hole_cards:
                hero_seat = 0
                if self._game_state_coordinator is not None:
                    hero_seat = self._game_state_coordinator._state.hero_seat
                detection_result.player_cards[hero_seat] = [
                    d.card for d in card_result.hole_cards
                ]

        # Feed detection result into game state coordinator
        if self._game_state_coordinator is not None:
            self._game_state_coordinator.process_frame(detection_result)

    def _on_new_hand_event(self, event: StateChangeEvent) -> None:
        """Handle a new hand event from the game state coordinator.

        Processes the completed hand through the stats aggregator.

        Args:
            event: The state change event containing the game state.
        """
        if self._stats_aggregator is not None:
            self._stats_aggregator.process_completed_hand(event.state)

    def _on_state_change_event(self, event: StateChangeEvent) -> None:
        """Handle game state changes by requesting solver advice and updating overlay.

        Args:
            event: The state change event containing the game state.
        """
        all_stats = {}
        if self._stats_aggregator is not None:
            all_stats = self._stats_aggregator.get_all_stats()

        if self._strategy_advisor is not None:
            self._strategy_advisor.get_advice_async(event.state, all_stats)

        # Update overlay with current stats
        self._update_overlay_stats(all_stats)

    def _on_advice_ready(self, advice: StrategyAdvice) -> None:
        """Handle solver advice results and update the overlay.

        Args:
            advice: Computed strategy advice.
        """
        if self._overlay is None:
            return

        # Format advice for display
        rec = advice.recommendation.name
        equity_pct = f"{advice.equity:.0%}"
        sizing = ""
        if advice.recommended_sizing is not None:
            sizing = f" ({advice.recommended_sizing:.0%} pot)"
        advice_text = f"{rec} | Equity: {equity_pct}{sizing}"
        overlay = self._overlay

        def _update_advice() -> None:
            overlay.set_text(advice_text)
            overlay.set_panel_content(PanelType.SOLVER, advice_text)

        self._run_on_main_thread(_update_advice)

    def _update_overlay_stats(
        self, all_stats: dict
    ) -> None:
        """Push current player stats to the overlay.

        Args:
            all_stats: Dict mapping player name to PlayerStats.
        """
        if self._overlay is None or self._stats_formatter is None:
            return

        stats_parts = []
        for player_name, player_stats in all_stats.items():
            formatted = self._stats_formatter.format_compact(player_stats)
            stats_parts.append(f"{player_name}: {formatted}")

        if stats_parts:
            stats_text = " | ".join(stats_parts)
            overlay = self._overlay

            def _update() -> None:
                overlay.set_text(stats_text)
                overlay.set_panel_content(PanelType.STATS, stats_text)

            self._run_on_main_thread(_update)

    def start(self) -> None:
        """Start all subsystems and begin processing.

        Raises:
            RuntimeError: If the application has not been initialized.
        """
        if self._state not in (AppState.INITIALIZED, AppState.PAUSED):
            raise RuntimeError(
                f"Cannot start from state {self._state.value}. "
                f"Call initialize() first."
            )

        logger.info("Starting Poker HUD...")

        if self._capture_pipeline is not None:
            self._capture_pipeline.start()

        if self._overlay is not None:
            self._overlay.show()

        self._state = AppState.RUNNING
        logger.info("Poker HUD is running")

    def stop(self) -> None:
        """Gracefully stop all subsystems in reverse initialization order."""
        if self._state == AppState.STOPPED:
            return

        logger.info("Stopping Poker HUD...")

        # Stop in reverse order of initialization
        if self._overlay is not None:
            self._overlay.close()

        if self._strategy_advisor is not None:
            self._strategy_advisor.shutdown()

        if self._capture_pipeline is not None:
            self._capture_pipeline.stop()

        if self._connection_manager is not None:
            self._connection_manager.close()

        self._state = AppState.STOPPED
        self._shutdown_event.set()
        logger.info("Poker HUD stopped")

    def pause(self) -> None:
        """Pause frame capture without shutting down subsystems."""
        if self._state != AppState.RUNNING:
            return

        if self._capture_pipeline is not None:
            self._capture_pipeline.pause()

        self._state = AppState.PAUSED
        logger.info("Poker HUD paused")

    def resume(self) -> None:
        """Resume frame capture after a pause."""
        if self._state != AppState.PAUSED:
            return

        if self._capture_pipeline is not None:
            self._capture_pipeline.resume()

        self._state = AppState.RUNNING
        logger.info("Poker HUD resumed")

    def install_signal_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers for graceful shutdown."""

        def _signal_handler(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, shutting down...", sig_name)
            self.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        logger.info("Signal handlers installed (SIGINT, SIGTERM)")

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> None:
        """Block until the application is stopped.

        Args:
            timeout: Maximum seconds to wait, or None for indefinite.
        """
        self._shutdown_event.wait(timeout=timeout)

    def _setup_logging(self) -> None:
        """Configure logging based on the debug setting."""
        level = (
            logging.DEBUG
            if self._debug or self._config.general.debug
            else logging.INFO
        )
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logger.info(
            "Logging configured at %s level",
            "DEBUG" if level == logging.DEBUG else "INFO",
        )

    @staticmethod
    def _run_on_main_thread(block: Callable[[], None]) -> None:
        """Schedule a callable to run on the main thread.

        AppKit requires all UI mutations to happen on the main thread.
        When called from a background thread (e.g. the capture pipeline),
        this dispatches via libdispatch. When already on the main thread
        it runs the block directly.
        """
        if threading.current_thread() is threading.main_thread():
            block()
            return

        try:
            from PyObjCTools.AppHelper import callAfter
            callAfter(block)
        except ImportError:
            # No AppKit available (headless mode) — run inline
            block()
