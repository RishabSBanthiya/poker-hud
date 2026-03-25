"""Integration tests for the cross-subsystem data pipeline.

Tests data flow between subsystems:
- Capture -> Detection -> Game State
- Game State -> Stats Aggregation
- Full hand lifecycle: Detection -> State -> Stats -> Advice
All tests mock the capture layer so no macOS screen recording is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.app import AppState, PokerHUDApp
from src.common.config import AppConfig, StatsConfig
from src.detection.card import Rank, Suit
from src.detection.detection_pipeline import PlayerInfo
from src.engine.game_state import ActionType, Street
from src.engine.game_state_coordinator import GameStateCoordinator
from src.engine.hand_history import HandState
from src.engine.hand_phase_tracker import HandPhaseTracker
from src.stats.connection_manager import ConnectionManager
from src.stats.hand_repository import HandRepository
from src.stats.player_stats_repository import (
    PlayerStats,
    PlayerStatsRepository,
)
from src.stats.stats_aggregator import StatsAggregator

from tests.conftest import (
    make_detected_card,
    make_detection_result,
    make_sample_hand_record,
)

# ---------------------------------------------------------------------------
# Capture -> Detection -> Game State flow
# ---------------------------------------------------------------------------


class TestCaptureToDetectionToGameState:
    """Test data flow from capture through detection into game state."""

    def test_frame_triggers_detection_and_state_update(self) -> None:
        """Detection results update game state via the coordinator."""
        hand_tracker = HandPhaseTracker()
        coordinator = GameStateCoordinator(hand_tracker)

        # Track state changes
        state_changes: list[HandState] = []
        coordinator.set_state_change_callback(
            lambda s: state_changes.append(s)
        )

        # Create a detection result with players
        detection = make_detection_result(
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(detection)

        # A new hand should have been started with our players
        assert coordinator.current_hand is not None
        player_names = {p.name for p in coordinator.current_hand.players}
        assert "Hero" in player_names
        assert "Villain" in player_names

    def test_community_cards_trigger_street_transitions(self) -> None:
        """Detection of community cards causes correct street transitions."""
        hand_tracker = HandPhaseTracker()
        coordinator = GameStateCoordinator(hand_tracker)

        coordinator.set_state_change_callback(lambda _: None)

        # Preflop: no community cards
        preflop_detection = make_detection_result(
            community_cards=[],
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(preflop_detection)
        assert coordinator.current_hand is not None
        assert coordinator.current_hand.street == Street.PREFLOP

        # Flop: 3 community cards
        flop_cards = [
            make_detected_card(Rank.ACE, Suit.HEARTS, x=200, y=100),
            make_detected_card(Rank.KING, Suit.DIAMONDS, x=270, y=100),
            make_detected_card(Rank.QUEEN, Suit.SPADES, x=340, y=100),
        ]
        flop_detection = make_detection_result(
            community_cards=flop_cards,
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(flop_detection)
        assert coordinator.current_hand.street == Street.FLOP

        # Turn: 4 community cards
        turn_cards = flop_cards + [
            make_detected_card(Rank.JACK, Suit.CLUBS, x=410, y=100),
        ]
        turn_detection = make_detection_result(
            community_cards=turn_cards,
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(turn_detection)
        assert coordinator.current_hand.street == Street.TURN

        # River: 5 community cards
        river_cards = turn_cards + [
            make_detected_card(Rank.TEN, Suit.HEARTS, x=480, y=100),
        ]
        river_detection = make_detection_result(
            community_cards=river_cards,
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(river_detection)
        assert coordinator.current_hand.street == Street.RIVER

    def test_hole_cards_assigned_to_hero(self) -> None:
        """Detected hole cards are assigned to the hero player."""
        hand_tracker = HandPhaseTracker()
        coordinator = GameStateCoordinator(hand_tracker)

        hole_cards = [
            make_detected_card(Rank.ACE, Suit.SPADES, x=350, y=450),
            make_detected_card(Rank.KING, Suit.SPADES, x=420, y=450),
        ]
        detection = make_detection_result(
            hole_cards=hole_cards,
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(detection)

        hand = coordinator.current_hand
        assert hand is not None
        hero = hand.get_player("Hero")
        assert hero is not None
        assert len(hero.hole_cards) == 2
        assert hero.hole_cards[0].rank == Rank.ACE
        assert hero.hole_cards[1].rank == Rank.KING


# ---------------------------------------------------------------------------
# Game State -> Stats Aggregation
# ---------------------------------------------------------------------------


class TestGameStateToStats:
    """Test data flow from game state to statistics aggregation."""

    def test_completed_hand_updates_stats(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """A completed hand record is stored and stats are calculated."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        record = make_sample_hand_record()
        aggregator.process_hand(record)

        assert hand_repo.count() == 1
        stored = hand_repo.get_by_id("test-001")
        assert stored is not None
        assert stored.pot == 10.0

        alice_stats = stats_repo.get("Alice")
        assert alice_stats is not None
        assert alice_stats.total_hands == 1

        bob_stats = stats_repo.get("Bob")
        assert bob_stats is not None
        assert bob_stats.total_hands == 1
        assert bob_stats.pfr_hands == 1

    def test_vpip_tracks_voluntary_money(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """VPIP counts only voluntary actions (not blind posts)."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        record = make_sample_hand_record()
        aggregator.process_hand(record)

        alice = stats_repo.get("Alice")
        assert alice is not None
        assert alice.vpip_hands == 1

        bob = stats_repo.get("Bob")
        assert bob is not None
        assert bob.vpip_hands == 1

        charlie = stats_repo.get("Charlie")
        assert charlie is not None
        assert charlie.vpip_hands == 0

    def test_multiple_hands_accumulate_stats(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Stats accumulate correctly across multiple hands."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        record1 = make_sample_hand_record(hand_id="h1")
        record2 = make_sample_hand_record(hand_id="h2")
        aggregator.process_hand(record1)
        aggregator.process_hand(record2)

        assert hand_repo.count() == 2

        bob = stats_repo.get("Bob")
        assert bob is not None
        assert bob.total_hands == 2
        assert bob.pfr_hands == 2

    def test_stats_callback_invoked_on_hand_complete(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """The stats update callback fires after processing a hand."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        updates: list[dict[str, PlayerStats]] = []
        aggregator.set_stats_update_callback(
            lambda stats: updates.append(stats)
        )

        record = make_sample_hand_record()
        aggregator.process_hand(record)

        assert len(updates) == 1
        assert "Alice" in updates[0]
        assert "Bob" in updates[0]

    def test_game_state_coordinator_to_stats_wiring(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Full wiring: coordinator completes hand -> stats aggregator processes it."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        hand_tracker = HandPhaseTracker()
        coordinator = GameStateCoordinator(hand_tracker)

        coordinator.set_hand_complete_callback(aggregator.process_hand)

        detection = make_detection_result(
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(detection)

        coordinator.record_action("Villain", ActionType.RAISE, 3.0)
        coordinator.record_action("Hero", ActionType.CALL, 3.0)

        record = coordinator.complete_hand(winner_name="Hero")
        assert record is not None

        assert hand_repo.count() == 1
        hero_stats = stats_repo.get("Hero")
        assert hero_stats is not None
        assert hero_stats.total_hands == 1


# ---------------------------------------------------------------------------
# Full hand lifecycle
# ---------------------------------------------------------------------------


class TestFullHandLifecycle:
    """Test the complete data pipeline from detection to advice."""

    def test_full_pipeline_detection_to_advice(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """End-to-end: detection -> game state -> stats."""
        hand_tracker = HandPhaseTracker()
        coordinator = GameStateCoordinator(hand_tracker)

        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        coordinator.set_hand_complete_callback(aggregator.process_hand)

        # Phase 1: Preflop with hero having pocket aces
        hole_cards = [
            make_detected_card(Rank.ACE, Suit.SPADES, x=350, y=450),
            make_detected_card(Rank.ACE, Suit.HEARTS, x=420, y=450),
        ]
        detection = make_detection_result(
            hole_cards=hole_cards,
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(detection)

        hand = coordinator.current_hand
        assert hand is not None
        hero = hand.get_player("Hero")
        assert hero is not None
        assert len(hero.hole_cards) == 2

        # Phase 2: Actions
        coordinator.record_action("Villain", ActionType.RAISE, 3.0)
        coordinator.record_action("Hero", ActionType.CALL, 3.0)

        # Phase 3: Flop
        flop_cards = [
            make_detected_card(Rank.ACE, Suit.DIAMONDS, x=200, y=100),
            make_detected_card(Rank.KING, Suit.CLUBS, x=270, y=100),
            make_detected_card(Rank.TWO, Suit.HEARTS, x=340, y=100),
        ]
        flop_detection = make_detection_result(
            community_cards=flop_cards,
            hole_cards=hole_cards,
            players=[
                PlayerInfo(seat_index=0, name="Hero", is_hero=True),
                PlayerInfo(seat_index=1, name="Villain"),
            ],
        )
        coordinator.process_detection(flop_detection)
        assert coordinator.current_hand.street == Street.FLOP

        # Phase 4: Complete
        record = coordinator.complete_hand(winner_name="Hero")
        assert record is not None

        hero_stats = stats_repo.get("Hero")
        assert hero_stats is not None
        assert hero_stats.total_hands == 1

    def test_new_hand_starts_after_completion(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """After completing a hand, a new detection starts a fresh hand."""
        hand_tracker = HandPhaseTracker()
        coordinator = GameStateCoordinator(hand_tracker)

        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)
        coordinator.set_hand_complete_callback(aggregator.process_hand)

        # Hand 1
        detection1 = make_detection_result(
            players=[
                PlayerInfo(seat_index=0, name="Alice", is_hero=True),
                PlayerInfo(seat_index=1, name="Bob"),
            ],
        )
        coordinator.process_detection(detection1)
        hand1_id = coordinator.current_hand.hand_id
        coordinator.complete_hand(winner_name="Alice")

        # Hand 2
        detection2 = make_detection_result(
            players=[
                PlayerInfo(seat_index=0, name="Alice", is_hero=True),
                PlayerInfo(seat_index=1, name="Bob"),
            ],
        )
        coordinator.process_detection(detection2)
        hand2_id = coordinator.current_hand.hand_id

        assert hand1_id != hand2_id
        assert hand_repo.count() == 1

    def test_stats_formatted_for_overlay(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Stats aggregator produces formatted HUD strings."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo, min_hands=2)

        record = make_sample_hand_record()
        aggregator.process_hand(record)

        hud_text = aggregator.format_player_hud("Bob")
        assert "1 hands" in hud_text

        record2 = make_sample_hand_record(hand_id="h2")
        aggregator.process_hand(record2)

        hud_text = aggregator.format_player_hud("Bob")
        assert "VPIP:" in hud_text
        assert "PFR:" in hud_text

    def test_action_counts_tracked_correctly(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Action counters (bets, raises, calls, folds) are tallied."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        record = make_sample_hand_record()
        aggregator.process_hand(record)

        bob = stats_repo.get("Bob")
        assert bob is not None
        assert bob.total_raises == 1
        assert bob.total_calls == 1

        alice = stats_repo.get("Alice")
        assert alice is not None
        assert alice.total_calls == 1
        assert alice.total_bets == 1


# ---------------------------------------------------------------------------
# App integration (mocked capture)
# ---------------------------------------------------------------------------


class TestAppIntegration:
    """Test the PokerHUDApp wiring with mocked capture layer."""

    def test_app_initializes_all_subsystems(self) -> None:
        """App initialize() creates and wires all subsystems."""
        config = AppConfig(
            stats=StatsConfig(db_path=":memory:")
        )
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()

        assert app.state == AppState.INITIALIZED
        assert app.capture_pipeline is not None
        assert app.game_state_coordinator is not None
        assert app.stats_aggregator is not None
        assert app.strategy_advisor is not None
        assert app.connection_manager is not None

        app.stop()
        assert app.state == AppState.STOPPED

    def test_app_pause_resume_cycle(self) -> None:
        """App can be paused and resumed."""
        config = AppConfig(
            stats=StatsConfig(db_path=":memory:")
        )
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()
        app.start()
        assert app.state == AppState.RUNNING

        app.pause()
        assert app.state == AppState.PAUSED

        app.resume()
        assert app.state == AppState.RUNNING

        app.stop()
        assert app.state == AppState.STOPPED

    def test_app_start_without_initialize_raises(self) -> None:
        """Starting without initializing raises RuntimeError."""
        app = PokerHUDApp(enable_overlay=False)
        with pytest.raises(RuntimeError, match="Cannot start"):
            app.start()

    def test_app_double_stop_is_safe(self) -> None:
        """Calling stop() twice does not raise."""
        config = AppConfig(
            stats=StatsConfig(db_path=":memory:")
        )
        app = PokerHUDApp(config=config, enable_overlay=False)
        app.initialize()
        app.stop()
        app.stop()
        assert app.state == AppState.STOPPED
