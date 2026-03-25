"""Unit tests for src.stats.aggregator module."""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

from src.engine.game_state import (
    ActionType,
    GameState,
    Player,
    PlayerAction,
    Street,
)
from src.stats.aggregator import RunningStatCounters, StatsAggregator
from src.stats.calculations import ConfidenceLevel

# -----------------------------------------------------------------------
# Helpers — build realistic GameState objects
# -----------------------------------------------------------------------


def _make_players(names: list[str], stack: float = 100.0) -> list[Player]:
    """Create a list of players with sequential seat numbers."""
    return [
        Player(name=n, seat_number=i, stack_size=stack)
        for i, n in enumerate(names)
    ]


def _gs(
    players: list[Player] | None = None,
    sb: float = 1.0,
    bb: float = 2.0,
    hand_id: str = "h1",
) -> GameState:
    """Build a minimal GameState for testing."""
    if players is None:
        players = _make_players(["Alice", "Bob", "Charlie"])
    return GameState(
        table_name="Test",
        small_blind=sb,
        big_blind=bb,
        players=players,
        hand_id=hand_id,
    )


def _post_blinds(gs: GameState, sb_seat: int = 0, bb_seat: int = 1) -> None:
    """Record blind posts on the game state."""
    gs.record_action(
        sb_seat, PlayerAction(ActionType.POST_BLIND, gs.small_blind, Street.PREFLOP)
    )
    gs.record_action(
        bb_seat, PlayerAction(ActionType.POST_BLIND, gs.big_blind, Street.PREFLOP)
    )


def _build_simple_hand_fold_preflop() -> GameState:
    """Hand: Alice SB, Bob BB, Charlie folds, Alice calls, Bob checks."""
    players = _make_players(["Alice", "Bob", "Charlie"])
    game = _gs(players, hand_id="simple-fold")
    _post_blinds(game, sb_seat=0, bb_seat=1)
    # Charlie folds
    game.record_action(2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))
    # Alice calls
    game.record_action(0, PlayerAction(ActionType.CALL, 1.0, Street.PREFLOP))
    # Bob checks
    game.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.PREFLOP))
    return game


def _build_hand_with_flop_and_cbet() -> GameState:
    """Hand: Alice raises preflop, Bob calls, Charlie folds.
    Flop: Alice bets (cbet), Bob calls.  Turn+River check through to showdown.
    """
    players = _make_players(["Alice", "Bob", "Charlie"])
    game = _gs(players, hand_id="cbet-hand")
    _post_blinds(game, sb_seat=0, bb_seat=1)

    # Preflop
    game.record_action(2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))
    game.record_action(0, PlayerAction(ActionType.RAISE, 5.0, Street.PREFLOP))
    game.record_action(1, PlayerAction(ActionType.CALL, 4.0, Street.PREFLOP))

    # Flop
    game.advance_street()
    game.record_action(0, PlayerAction(ActionType.BET, 6.0, Street.FLOP))
    game.record_action(1, PlayerAction(ActionType.CALL, 6.0, Street.FLOP))

    # Turn
    game.advance_street()
    game.record_action(0, PlayerAction(ActionType.CHECK, 0.0, Street.TURN))
    game.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.TURN))

    # River
    game.advance_street()
    game.record_action(0, PlayerAction(ActionType.CHECK, 0.0, Street.RIVER))
    game.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.RIVER))

    # Showdown
    game.advance_street()
    return game


def _build_hand_with_3bet() -> GameState:
    """Hand: Alice raises, Bob 3-bets, Charlie folds, Alice calls."""
    players = _make_players(["Alice", "Bob", "Charlie"])
    game = _gs(players, hand_id="3bet-hand")
    _post_blinds(game, sb_seat=0, bb_seat=1)

    game.record_action(2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))
    game.record_action(0, PlayerAction(ActionType.RAISE, 5.0, Street.PREFLOP))
    game.record_action(1, PlayerAction(ActionType.RAISE, 15.0, Street.PREFLOP))
    game.record_action(0, PlayerAction(ActionType.CALL, 11.0, Street.PREFLOP))

    # Go to flop and showdown for WTSD tracking
    game.advance_street()
    game.record_action(0, PlayerAction(ActionType.CHECK, 0.0, Street.FLOP))
    game.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.FLOP))
    game.advance_street()
    game.record_action(0, PlayerAction(ActionType.CHECK, 0.0, Street.TURN))
    game.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.TURN))
    game.advance_street()
    game.record_action(0, PlayerAction(ActionType.CHECK, 0.0, Street.RIVER))
    game.record_action(1, PlayerAction(ActionType.CHECK, 0.0, Street.RIVER))
    game.advance_street()
    return game


def _build_hand_fold_to_3bet() -> GameState:
    """Hand: Alice raises, Bob 3-bets, Alice folds."""
    players = _make_players(["Alice", "Bob", "Charlie"])
    game = _gs(players, hand_id="fold-to-3bet")
    _post_blinds(game, sb_seat=0, bb_seat=1)

    game.record_action(2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))
    game.record_action(0, PlayerAction(ActionType.RAISE, 5.0, Street.PREFLOP))
    game.record_action(1, PlayerAction(ActionType.RAISE, 15.0, Street.PREFLOP))
    game.record_action(0, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP))
    return game


# -----------------------------------------------------------------------
# Basic construction
# -----------------------------------------------------------------------


class TestStatsAggregatorInit:
    def test_empty_aggregator(self) -> None:
        agg = StatsAggregator()
        assert agg.get_all_stats() == {}

    def test_unknown_player(self) -> None:
        agg = StatsAggregator()
        stats = agg.get_player_stats("Nobody")
        assert stats.total_hands == 0
        assert stats.vpip == 0.0


# -----------------------------------------------------------------------
# process_completed_hand
# -----------------------------------------------------------------------


class TestProcessCompletedHand:
    def test_single_hand_updates_all_players(self) -> None:
        agg = StatsAggregator()
        game = _build_simple_hand_fold_preflop()
        agg.process_completed_hand(game)

        all_stats = agg.get_all_stats()
        assert "Alice" in all_stats
        assert "Bob" in all_stats
        assert "Charlie" in all_stats
        assert all(s.total_hands == 1 for s in all_stats.values())

    def test_idempotent_processing(self) -> None:
        """Processing the same hand twice should not double-count."""
        agg = StatsAggregator()
        game = _build_simple_hand_fold_preflop()
        agg.process_completed_hand(game)
        agg.process_completed_hand(game)

        stats = agg.get_player_stats("Alice")
        assert stats.total_hands == 1

    def test_multiple_hands(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())

        alice = agg.get_player_stats("Alice")
        assert alice.total_hands == 2


# -----------------------------------------------------------------------
# VPIP via aggregator
# -----------------------------------------------------------------------


class TestAggregatorVPIP:
    def test_fold_is_not_vpip(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        charlie = agg.get_player_stats("Charlie")
        assert charlie.vpip == 0.0

    def test_call_is_vpip(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        alice = agg.get_player_stats("Alice")
        assert alice.vpip == 100.0

    def test_check_in_bb_is_not_vpip(self) -> None:
        """BB checking is not voluntary -- should not count as VPIP."""
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        bob = agg.get_player_stats("Bob")
        assert bob.vpip == 0.0


# -----------------------------------------------------------------------
# PFR via aggregator
# -----------------------------------------------------------------------


class TestAggregatorPFR:
    def test_raise_counts_as_pfr(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        alice = agg.get_player_stats("Alice")
        assert alice.pfr == 100.0

    def test_call_not_pfr(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        bob = agg.get_player_stats("Bob")
        assert bob.pfr == 0.0


# -----------------------------------------------------------------------
# C-Bet via aggregator
# -----------------------------------------------------------------------


class TestAggregatorCBet:
    def test_cbet_detected(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        alice = agg.get_player_stats("Alice")
        assert alice.cbet_pct == 100.0

    def test_non_raiser_no_cbet_opportunity(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        bob = agg.get_player_stats("Bob")
        assert bob.cbet_pct == 0.0

    def test_no_flop_no_cbet(self) -> None:
        """Hand ended preflop -- raiser has no cbet opportunity."""
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_fold_to_3bet())
        # Alice raised but hand ended preflop
        alice_counters = agg.get_counters("Alice")
        assert alice_counters is not None
        assert alice_counters.cbet_opportunities == 0


# -----------------------------------------------------------------------
# 3-Bet via aggregator
# -----------------------------------------------------------------------


class TestAggregatorThreeBet:
    def test_3bet_detected(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_3bet())
        bob = agg.get_player_stats("Bob")
        # Bob 3-bet (raised after Alice's raise)
        counters = agg.get_counters("Bob")
        assert counters is not None
        assert counters.three_bet_count == 1
        assert counters.three_bet_opportunities == 1
        assert bob.three_bet_pct == 100.0

    def test_no_3bet_opportunity_for_opener(self) -> None:
        """Alice opened, so she didn't face a raise before her first action."""
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_3bet())
        counters = agg.get_counters("Alice")
        assert counters is not None
        # Alice's first voluntary action was the open-raise; no raise before it
        assert counters.three_bet_opportunities == 0


# -----------------------------------------------------------------------
# Fold to 3-Bet via aggregator
# -----------------------------------------------------------------------


class TestAggregatorFoldTo3Bet:
    def test_fold_to_3bet(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_fold_to_3bet())
        alice = agg.get_player_stats("Alice")
        counters = agg.get_counters("Alice")
        assert counters is not None
        assert counters.fold_to_3bet_opportunities == 1
        assert counters.fold_to_3bet_count == 1
        assert alice.fold_to_three_bet == 100.0

    def test_call_3bet_not_fold(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_3bet())
        alice = agg.get_player_stats("Alice")
        counters = agg.get_counters("Alice")
        assert counters is not None
        assert counters.fold_to_3bet_opportunities == 1
        assert counters.fold_to_3bet_count == 0
        assert alice.fold_to_three_bet == 0.0


# -----------------------------------------------------------------------
# WTSD via aggregator
# -----------------------------------------------------------------------


class TestAggregatorWTSD:
    def test_showdown_reached(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        # Alice and Bob both saw flop and reached showdown
        alice = agg.get_player_stats("Alice")
        bob = agg.get_player_stats("Bob")
        assert alice.wtsd == 100.0
        assert bob.wtsd == 100.0

    def test_folded_preflop_no_wtsd(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        charlie = agg.get_player_stats("Charlie")
        # Charlie folded preflop, never saw flop
        counters = agg.get_counters("Charlie")
        assert counters is not None
        assert counters.saw_flop_count == 0
        assert charlie.wtsd == 0.0


# -----------------------------------------------------------------------
# Aggression Factor via aggregator
# -----------------------------------------------------------------------


class TestAggregatorAF:
    def test_aggressive_player(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        alice = agg.get_player_stats("Alice")
        # Alice: raise preflop (1) + bet flop (1) = 2 aggressive, 0 calls (blind call
        # is actually CALL action in the hand but she used RAISE)
        # Actually: preflop RAISE + flop BET = 2 aggressive, 0 passive
        assert alice.aggression_factor == float("inf")

    def test_passive_player(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())
        bob = agg.get_player_stats("Bob")
        # Bob: preflop CALL + flop CALL = 0 aggressive, 2 passive -> AF = 0
        assert bob.aggression_factor == 0.0


# -----------------------------------------------------------------------
# get_table_stats
# -----------------------------------------------------------------------


class TestGetTableStats:
    def test_returns_only_requested_players(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        table = agg.get_table_stats(["Alice", "Bob"])
        assert "Alice" in table
        assert "Bob" in table
        assert "Charlie" not in table

    def test_unknown_player_gets_empty_stats(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        table = agg.get_table_stats(["Alice", "Unknown"])
        assert table["Unknown"].total_hands == 0

    def test_empty_seat_list(self) -> None:
        agg = StatsAggregator()
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        assert agg.get_table_stats([]) == {}


# -----------------------------------------------------------------------
# Persistence callback
# -----------------------------------------------------------------------


class TestPersistenceCallback:
    def test_callback_called(self) -> None:
        callback = MagicMock()
        agg = StatsAggregator(persistence_callback=callback)
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        callback.assert_called_once()
        # Arg should be a dict of counters
        arg = callback.call_args[0][0]
        assert isinstance(arg, dict)
        assert "Alice" in arg

    def test_callback_exception_does_not_propagate(self) -> None:
        def bad_callback(counters: Dict) -> None:
            raise RuntimeError("disk full")

        agg = StatsAggregator(persistence_callback=bad_callback)
        # Should not raise
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        # Stats should still be updated
        assert agg.get_player_stats("Alice").total_hands == 1


# -----------------------------------------------------------------------
# Thread safety (basic smoke test)
# -----------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_reads_and_writes(self) -> None:
        """Smoke test: process hands and read stats from multiple threads."""
        import threading

        agg = StatsAggregator()
        errors: list[Exception] = []

        def process_hands() -> None:
            try:
                for i in range(20):
                    players = _make_players(["Alice", "Bob", "Charlie"])
                    tid = threading.current_thread().name
                    game = _gs(players, hand_id=f"thread-{tid}-{i}")
                    _post_blinds(game)
                    game.record_action(
                        2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP)
                    )
                    game.record_action(
                        0, PlayerAction(ActionType.CALL, 1.0, Street.PREFLOP)
                    )
                    game.record_action(
                        1, PlayerAction(ActionType.CHECK, 0.0, Street.PREFLOP)
                    )
                    agg.process_completed_hand(game)
            except Exception as e:
                errors.append(e)

        def read_stats() -> None:
            try:
                for _ in range(50):
                    agg.get_player_stats("Alice")
                    agg.get_all_stats()
                    agg.get_table_stats(["Alice", "Bob"])
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(3):
            t = threading.Thread(target=process_hands, name=f"writer-{i}")
            threads.append(t)
        for i in range(3):
            t = threading.Thread(target=read_stats, name=f"reader-{i}")
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


# -----------------------------------------------------------------------
# Multi-hand scenario
# -----------------------------------------------------------------------


class TestMultiHandScenario:
    """Process several hands and verify accumulated stats."""

    def test_accumulated_stats(self) -> None:
        agg = StatsAggregator()

        # Hand 1: Alice calls, Bob checks BB, Charlie folds
        agg.process_completed_hand(_build_simple_hand_fold_preflop())

        # Hand 2: Alice raises + cbets, Bob calls, Charlie folds
        agg.process_completed_hand(_build_hand_with_flop_and_cbet())

        alice = agg.get_player_stats("Alice")
        assert alice.total_hands == 2
        # Hand 1: VPIP (call), Hand 2: VPIP (raise) => 100%
        assert alice.vpip == 100.0
        # Hand 1: no raise, Hand 2: raise => 50%
        assert alice.pfr == 50.0

        charlie = agg.get_player_stats("Charlie")
        assert charlie.total_hands == 2
        assert charlie.vpip == 0.0
        assert charlie.pfr == 0.0

    def test_confidence_increases_with_hands(self) -> None:
        agg = StatsAggregator()
        # Process 1 hand
        agg.process_completed_hand(_build_simple_hand_fold_preflop())
        assert agg.get_player_stats("Alice").confidence == ConfidenceLevel.VERY_LOW

        # Process enough hands to reach LOW confidence (10+)
        for i in range(12):
            players = _make_players(["Alice", "Bob", "Charlie"])
            game = _gs(players, hand_id=f"bulk-{i}")
            _post_blinds(game)
            game.record_action(
                2, PlayerAction(ActionType.FOLD, 0.0, Street.PREFLOP)
            )
            game.record_action(
                0, PlayerAction(ActionType.CALL, 1.0, Street.PREFLOP)
            )
            game.record_action(
                1, PlayerAction(ActionType.CHECK, 0.0, Street.PREFLOP)
            )
            agg.process_completed_hand(game)

        alice = agg.get_player_stats("Alice")
        assert alice.confidence == ConfidenceLevel.LOW
        assert alice.total_hands == 13  # 1 + 12


# -----------------------------------------------------------------------
# RunningStatCounters
# -----------------------------------------------------------------------


class TestRunningStatCounters:
    def test_defaults(self) -> None:
        c = RunningStatCounters()
        assert c.total_hands == 0
        assert c.vpip_count == 0
        assert c.processed_hand_ids == set()
