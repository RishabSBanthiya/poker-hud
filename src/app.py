"""Main application class for the Poker HUD.

Orchestrates initialization, wiring, and lifecycle management
of all subsystems: capture, detection, engine, stats, solver, overlay.
"""

from __future__ import annotations

import logging
import signal
import threading
from enum import Enum
from typing import Optional

import numpy as np

from src.capture.capture_pipeline import CapturePipeline
from src.capture.frame_poller import FramePoller
from src.capture.screen_capture import ScreenCapture
from src.capture.window_detector import WindowDetector
from src.common.config import AppConfig
from src.detection.card_recognition import CardRecognitionPipeline
from src.detection.detection_pipeline import DetectionPipeline, DetectionResult
from src.detection.ocr_engine import OCREngine
from src.detection.player_identifier import PlayerIdentifier
from src.engine.game_state import HandState
from src.engine.game_state_coordinator import GameStateCoordinator
from src.engine.hand_history import HandRecord
from src.engine.hand_phase_tracker import HandPhaseTracker
from src.solver.equity_calculator import EquityCalculator
from src.solver.strategy_advisor import StrategyAdvisorCoordinator
from src.stats.connection_manager import ConnectionManager
from src.stats.hand_repository import HandRepository
from src.stats.player_stats_repository import PlayerStatsRepository
from src.stats.stats_aggregator import StatsAggregator

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
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        enable_overlay: bool = True,
    ) -> None:
        self._config = config or AppConfig()
        self._enable_overlay = enable_overlay
        self._state = AppState.CREATED
        self._shutdown_event = threading.Event()

        # Subsystem references (initialized in initialize())
        self._connection_manager: Optional[ConnectionManager] = None
        self._capture_pipeline: Optional[CapturePipeline] = None
        self._detection_pipeline: Optional[DetectionPipeline] = None
        self._game_state_coordinator: Optional[GameStateCoordinator] = None
        self._stats_aggregator: Optional[StatsAggregator] = None
        self._strategy_advisor: Optional[StrategyAdvisorCoordinator] = None
        self._overlay_text_callback: Optional[
            callable  # type: ignore[type-arg]
        ] = None

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
    def detection_pipeline(self) -> Optional[DetectionPipeline]:
        """Detection pipeline instance."""
        return self._detection_pipeline

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

    def initialize(self) -> None:
        """Initialize all subsystems in dependency order.

        1. Configuration (already loaded)
        2. Logging
        3. Database / connection manager
        4. Capture pipeline (window detector, screen capture, frame poller)
        5. Detection pipeline (card recognition, OCR, player ID)
        6. Game state engine (coordinator, hand tracker)
        7. Stats aggregator (connected to database)
        8. Strategy advisor
        9. Overlay (if enabled)

        Then wires subsystems together via callbacks.
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
        self._connection_manager = ConnectionManager(
            db_path=self._config.stats.db_path
        )
        self._connection_manager.initialize()

        # 3. Capture pipeline
        window_detector = WindowDetector(
            title_pattern=self._config.capture.window_title_pattern
        )
        screen_capture = ScreenCapture()
        frame_poller = FramePoller(
            screen_capture=screen_capture,
            poll_interval_ms=self._config.capture.poll_interval_ms,
            change_threshold=self._config.capture.frame_change_threshold,
        )
        self._capture_pipeline = CapturePipeline(
            window_detector=window_detector,
            screen_capture=screen_capture,
            frame_poller=frame_poller,
        )

        # 4. Detection pipeline
        card_recognition = CardRecognitionPipeline(
            template_dir=self._config.detection.template_dir,
            confidence_threshold=self._config.detection.confidence_threshold,
        )
        ocr_engine = OCREngine()
        player_identifier = PlayerIdentifier(ocr_engine=ocr_engine)
        self._detection_pipeline = DetectionPipeline(
            card_recognition=card_recognition,
            ocr_engine=ocr_engine,
            player_identifier=player_identifier,
        )
        self._detection_pipeline.initialize()

        # 5. Game state engine
        hand_tracker = HandPhaseTracker()
        self._game_state_coordinator = GameStateCoordinator(
            hand_phase_tracker=hand_tracker
        )

        # 6. Stats aggregator
        hand_repo = HandRepository(self._connection_manager)
        stats_repo = PlayerStatsRepository(self._connection_manager)
        self._stats_aggregator = StatsAggregator(
            hand_repo=hand_repo,
            stats_repo=stats_repo,
            min_hands=self._config.stats.min_hands_for_stats,
        )

        # 7. Strategy advisor
        equity_calc = EquityCalculator(
            num_simulations=self._config.solver.equity_simulations
        )
        self._strategy_advisor = StrategyAdvisorCoordinator(
            equity_calculator=equity_calc
        )

        # 8. Wire subsystems together
        self._wire_subsystems()

        self._state = AppState.INITIALIZED
        logger.info("Poker HUD application initialized successfully")

    def _wire_subsystems(self) -> None:
        """Connect subsystems via callbacks to form the data pipeline.

        Data flow:
            Capture -> Detection -> Game State -> Stats + Solver -> Overlay
        """
        # Capture -> Detection: frame callback
        if self._capture_pipeline and self._detection_pipeline:
            self._capture_pipeline.set_frame_callback(
                self._on_frame_captured
            )

        # Detection -> Game State: detection result callback
        if self._detection_pipeline and self._game_state_coordinator:
            self._detection_pipeline.set_result_callback(
                self._on_detection_result
            )

        # Game State -> Stats: completed hand callback
        if self._game_state_coordinator and self._stats_aggregator:
            self._game_state_coordinator.set_hand_complete_callback(
                self._on_hand_complete
            )

        # Game State -> Solver: state change callback
        if self._game_state_coordinator and self._strategy_advisor:
            self._game_state_coordinator.set_state_change_callback(
                self._on_state_change
            )

    def _on_frame_captured(self, frame: np.ndarray) -> None:
        """Handle a newly captured frame by passing it to detection.

        Args:
            frame: BGR numpy array of the captured frame.
        """
        if self._detection_pipeline is not None:
            self._detection_pipeline.process_frame(frame)

    def _on_detection_result(self, result: DetectionResult) -> None:
        """Handle detection results by updating game state.

        Args:
            result: Combined detection result.
        """
        if self._game_state_coordinator is not None:
            self._game_state_coordinator.process_detection(result)

    def _on_hand_complete(self, record: HandRecord) -> None:
        """Handle a completed hand by updating stats.

        Args:
            record: The completed hand record.
        """
        if self._stats_aggregator is not None:
            self._stats_aggregator.process_hand(record)

    def _on_state_change(self, hand_state: HandState) -> None:
        """Handle game state changes by requesting solver advice.

        Args:
            hand_state: Updated hand state.
        """
        if self._strategy_advisor is not None:
            self._strategy_advisor.analyze_state(hand_state)

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

        self._state = AppState.RUNNING
        logger.info("Poker HUD is running")

    def stop(self) -> None:
        """Gracefully stop all subsystems in reverse initialization order."""
        if self._state == AppState.STOPPED:
            return

        logger.info("Stopping Poker HUD...")

        # Stop in reverse order of initialization
        if self._capture_pipeline is not None:
            self._capture_pipeline.stop()

        if self._detection_pipeline is not None:
            self._detection_pipeline.shutdown()

        if self._connection_manager is not None:
            self._connection_manager.close_all()

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
        level = logging.DEBUG if self._config.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logger.info(
            "Logging configured at %s level",
            "DEBUG" if self._config.debug else "INFO",
        )
