"""Tests for src.solver.advisor_coordinator."""

from __future__ import annotations

import threading

import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    ActionType,
    GameState,
    Player,
    PlayerAction,
    Position,
    Street,
)
from src.solver.advisor_coordinator import (
    ActionRecommendation,
    StrategyAdvice,
    StrategyAdvisorCoordinator,
    _estimate_hand_equity,
    _game_state_key,
)
from src.solver.range_estimator import HandRange
from src.stats.calculations import PlayerStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game_state(
    hero_cards: list[Card] | None = None,
    community_cards: list[Card] | None = None,
    street: Street = Street.PREFLOP,
    pot: float = 10.0,
    villain_actions: list[PlayerAction] | None = None,
) -> GameState:
    """Build a minimal GameState for testing."""
    hero = Player(
        name="Hero",
        seat_number=0,
        position=Position.BTN,
        stack_size=100.0,
        hole_cards=hero_cards,
    )
    villain = Player(
        name="Villain",
        seat_number=1,
        position=Position.BB,
        stack_size=100.0,
    )
    if villain_actions:
        for a in villain_actions:
            a.player_name = "Villain"
            villain.actions.append(a)

    gs = GameState(
        players=[hero, villain],
        hero_seat=0,
        current_street=street,
        community_cards=community_cards or [],
        pot_size=pot,
    )
    return gs


def _make_hero_cards(rank1: Rank, suit1: Suit, rank2: Rank, suit2: Suit) -> list[Card]:
    return [Card(rank1, suit1), Card(rank2, suit2)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def advisor() -> StrategyAdvisorCoordinator:
    coordinator = StrategyAdvisorCoordinator()
    yield coordinator
    coordinator.shutdown()


@pytest.fixture
def tight_stats() -> dict[str, PlayerStats]:
    return {
        "Villain": PlayerStats(
            vpip=15.0, pfr=12.0, three_bet_pct=4.0,
            aggression_factor=3.0, cbet_pct=75.0, total_hands=200,
        )
    }


@pytest.fixture
def loose_stats() -> dict[str, PlayerStats]:
    return {
        "Villain": PlayerStats(
            vpip=45.0, pfr=8.0, three_bet_pct=2.0,
            aggression_factor=0.8, cbet_pct=40.0, total_hands=150,
        )
    }


# ---------------------------------------------------------------------------
# StrategyAdvice dataclass tests
# ---------------------------------------------------------------------------


class TestStrategyAdvice:
    """Tests for the StrategyAdvice dataclass."""

    def test_default_values(self) -> None:
        advice = StrategyAdvice()
        assert advice.preflop_range_position is False
        assert advice.equity == 0.0
        assert advice.recommendation == ActionRecommendation.FOLD
        assert advice.opponent_ranges == {}
        assert advice.reasoning == []

    def test_custom_values(self) -> None:
        advice = StrategyAdvice(
            preflop_range_position=True,
            equity=0.65,
            recommendation=ActionRecommendation.RAISE,
            recommended_sizing=0.75,
            reasoning=["Strong hand"],
        )
        assert advice.equity == 0.65
        assert advice.recommendation == ActionRecommendation.RAISE
        assert advice.recommended_sizing == 0.75


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


class TestGameStateKey:
    """Tests for cache key generation."""

    def test_same_state_same_key(self) -> None:
        gs = _make_game_state()
        assert _game_state_key(gs) == _game_state_key(gs)

    def test_different_street_different_key(self) -> None:
        gs1 = _make_game_state(street=Street.PREFLOP)
        gs2 = _make_game_state(street=Street.FLOP)
        gs2.hand_id = gs1.hand_id
        assert _game_state_key(gs1) != _game_state_key(gs2)

    def test_different_pot_different_key(self) -> None:
        gs1 = _make_game_state(pot=10.0)
        gs2 = _make_game_state(pot=20.0)
        gs2.hand_id = gs1.hand_id
        assert _game_state_key(gs1) != _game_state_key(gs2)


# ---------------------------------------------------------------------------
# Equity estimation tests
# ---------------------------------------------------------------------------


class TestEstimateHandEquity:
    """Tests for the heuristic equity estimator."""

    def test_pocket_aces_high_equity(self) -> None:
        cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        eq = _estimate_hand_equity(cards, [], HandRange.full_range())
        assert eq > 0.70

    def test_low_cards_lower_equity(self) -> None:
        cards = _make_hero_cards(Rank.TWO, Suit.SPADES, Rank.SEVEN, Suit.HEARTS)
        eq = _estimate_hand_equity(cards, [], HandRange.full_range())
        assert eq < 0.50

    def test_equity_against_tight_range_is_lower(self) -> None:
        cards = _make_hero_cards(Rank.JACK, Suit.SPADES, Rank.TEN, Suit.HEARTS)
        tight_range = HandRange.empty_range()
        tight_range.set_weight(Rank.ACE, Rank.ACE, suited=False, weight=1.0)
        tight_range.set_weight(Rank.KING, Rank.KING, suited=False, weight=1.0)

        wide_range = HandRange.full_range()

        eq_tight = _estimate_hand_equity(cards, [], tight_range)
        eq_wide = _estimate_hand_equity(cards, [], wide_range)
        assert eq_tight < eq_wide

    def test_board_hit_boosts_equity(self) -> None:
        cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.KING, Suit.HEARTS)
        board_miss = [
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
        ]
        board_hit = [
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
        ]
        eq_miss = _estimate_hand_equity(cards, board_miss, HandRange.full_range())
        eq_hit = _estimate_hand_equity(cards, board_hit, HandRange.full_range())
        assert eq_hit > eq_miss

    def test_no_hole_cards_returns_default(self) -> None:
        eq = _estimate_hand_equity([], [], HandRange.full_range())
        assert eq == pytest.approx(0.5)

    def test_suited_bonus(self) -> None:
        suited = _make_hero_cards(Rank.KING, Suit.SPADES, Rank.QUEEN, Suit.SPADES)
        offsuit = _make_hero_cards(Rank.KING, Suit.SPADES, Rank.QUEEN, Suit.HEARTS)
        eq_suited = _estimate_hand_equity(suited, [], HandRange.full_range())
        eq_offsuit = _estimate_hand_equity(offsuit, [], HandRange.full_range())
        assert eq_suited > eq_offsuit


# ---------------------------------------------------------------------------
# StrategyAdvisorCoordinator: synchronous get_advice
# ---------------------------------------------------------------------------


class TestGetAdvice:
    """Tests for synchronous advice generation."""

    def test_hero_not_found_returns_reasoning(
        self, advisor: StrategyAdvisorCoordinator
    ) -> None:
        gs = GameState(players=[], hero_seat=99)
        advice = advisor.get_advice(gs, {})
        assert "Hero not found" in advice.reasoning[0]

    def test_strong_hand_recommends_raise(
        self, advisor: StrategyAdvisorCoordinator, loose_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards, pot=20.0)
        advice = advisor.get_advice(gs, loose_stats)
        assert advice.equity > 0.5
        assert advice.preflop_range_position is True
        # Strong hand should not be FOLD
        assert advice.recommendation != ActionRecommendation.FOLD

    def test_weak_hand_outside_range_recommends_fold(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.TWO, Suit.SPADES, Rank.SEVEN, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advice = advisor.get_advice(gs, tight_stats)
        assert advice.preflop_range_position is False
        assert advice.recommendation == ActionRecommendation.FOLD

    def test_unknown_hero_cards_defaults_equity(
        self, advisor: StrategyAdvisorCoordinator, loose_stats: dict[str, PlayerStats]
    ) -> None:
        gs = _make_game_state(hero_cards=None)
        advice = advisor.get_advice(gs, loose_stats)
        msg = advice.reasoning[1].lower()
        assert "unknown" in msg or "default" in msg

    def test_opponent_ranges_populated(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.KING, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advice = advisor.get_advice(gs, tight_stats)
        assert "Villain" in advice.opponent_ranges
        assert isinstance(advice.opponent_ranges["Villain"], HandRange)

    def test_reasoning_not_empty(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.KING, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advice = advisor.get_advice(gs, tight_stats)
        assert len(advice.reasoning) >= 2

    def test_postflop_advice(
        self, advisor: StrategyAdvisorCoordinator, loose_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.KING, Suit.HEARTS)
        board = [
            Card(Rank.ACE, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.TWO, Suit.HEARTS),
        ]
        gs = _make_game_state(
            hero_cards=hero_cards,
            community_cards=board,
            street=Street.FLOP,
            pot=30.0,
        )
        advice = advisor.get_advice(gs, loose_stats)
        # Hit top pair: should have decent equity
        assert advice.equity > 0.4

    def test_villain_actions_affect_range(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.QUEEN, Suit.SPADES, Rank.QUEEN, Suit.HEARTS)
        villain_actions = [
            PlayerAction(
                action_type=ActionType.RAISE, amount=6.0, street=Street.PREFLOP
            ),
        ]
        gs = _make_game_state(
            hero_cards=hero_cards,
            villain_actions=villain_actions,
        )
        advice = advisor.get_advice(gs, tight_stats)
        # Tight villain raised: their range should be narrow
        villain_range = advice.opponent_ranges.get("Villain")
        assert villain_range is not None
        assert villain_range.range_pct() < 30.0


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestAdviceCache:
    """Tests for advice caching."""

    def test_same_state_returns_cached(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advice1 = advisor.get_advice(gs, tight_stats)
        advice2 = advisor.get_advice(gs, tight_stats)
        assert advice1.equity == advice2.equity

    def test_clear_cache(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advisor.get_advice(gs, tight_stats)
        advisor.clear_cache()
        # Cache should be empty now -- no error on re-compute
        advice = advisor.get_advice(gs, tight_stats)
        assert advice.equity > 0


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsyncAdvice:
    """Tests for async advice computation."""

    def test_async_returns_future(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        future = advisor.get_advice_async(gs, tight_stats)
        result = future.result(timeout=5.0)
        assert isinstance(result, StrategyAdvice)
        assert result.equity > 0

    def test_callback_invoked(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        results: list[StrategyAdvice] = []
        event = threading.Event()

        def on_ready(advice: StrategyAdvice) -> None:
            results.append(advice)
            event.set()

        advisor.on_advice_ready(on_ready)
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advisor.get_advice_async(gs, tight_stats)

        assert event.wait(timeout=5.0), "Callback was not invoked within timeout"
        assert len(results) == 1
        assert isinstance(results[0], StrategyAdvice)

    def test_callback_error_does_not_break(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        def bad_callback(advice: StrategyAdvice) -> None:
            raise RuntimeError("intentional test error")

        advisor.on_advice_ready(bad_callback)
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        future = advisor.get_advice_async(gs, tight_stats)
        # Should not raise despite callback error
        result = future.result(timeout=5.0)
        assert isinstance(result, StrategyAdvice)


# ---------------------------------------------------------------------------
# Latency tracking tests
# ---------------------------------------------------------------------------


class TestLatencyTracking:
    """Tests for performance monitoring integration."""

    def test_latency_summary_populated_after_call(
        self, advisor: StrategyAdvisorCoordinator, tight_stats: dict[str, PlayerStats]
    ) -> None:
        hero_cards = _make_hero_cards(Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS)
        gs = _make_game_state(hero_cards=hero_cards)
        advisor.get_advice(gs, tight_stats)

        summary = advisor.get_latency_summary()
        assert "advisor_full_pipeline" in summary
        assert summary["advisor_full_pipeline"] >= 0

    def test_latency_no_measurements_initially(
        self, advisor: StrategyAdvisorCoordinator
    ) -> None:
        summary = advisor.get_latency_summary()
        assert len(summary) == 0


# ---------------------------------------------------------------------------
# Preflop range position tests
# ---------------------------------------------------------------------------


class TestPreflopRangeCheck:
    """Tests for the preflop range position check."""

    def test_pocket_aces_in_range(self) -> None:
        cards = _make_hero_cards(
            Rank.ACE, Suit.SPADES, Rank.ACE, Suit.HEARTS
        )
        hero = Player(name="Hero", seat_number=0, hole_cards=cards)
        check = StrategyAdvisorCoordinator._check_preflop_range
        assert check(hero, Street.PREFLOP) is True

    def test_72o_not_in_range(self) -> None:
        cards = _make_hero_cards(
            Rank.SEVEN, Suit.SPADES, Rank.TWO, Suit.HEARTS
        )
        hero = Player(name="Hero", seat_number=0, hole_cards=cards)
        check = StrategyAdvisorCoordinator._check_preflop_range
        assert check(hero, Street.PREFLOP) is False

    def test_suited_ace_in_range(self) -> None:
        cards = _make_hero_cards(
            Rank.ACE, Suit.SPADES, Rank.FIVE, Suit.SPADES
        )
        hero = Player(name="Hero", seat_number=0, hole_cards=cards)
        check = StrategyAdvisorCoordinator._check_preflop_range
        assert check(hero, Street.PREFLOP) is True

    def test_no_hole_cards_not_in_range(self) -> None:
        hero = Player(name="Hero", seat_number=0, hole_cards=None)
        check = StrategyAdvisorCoordinator._check_preflop_range
        assert check(hero, Street.PREFLOP) is False

    def test_postflop_always_true(self) -> None:
        hero = Player(name="Hero", seat_number=0, hole_cards=None)
        check = StrategyAdvisorCoordinator._check_preflop_range
        assert check(hero, Street.FLOP) is True
