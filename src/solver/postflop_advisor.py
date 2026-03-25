"""Equity-based postflop advisor for the solver subsystem.

Combines pot odds analysis with hand equity to recommend postflop actions.
Includes draw detection and implied odds adjustments for drawing hands.

Usage:
    from src.solver.postflop_advisor import PostflopAdvisor
    from src.engine.game_state import GameState

    advisor = PostflopAdvisor()
    rec = advisor.get_recommendation(game_state, hero_hand)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.detection.card import Card, Suit
from src.engine.game_state import GameState, HandStrength, Street
from src.solver.equity import _RANK_VAL, EquityCalculator, evaluate_hand

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Action(Enum):
    """Postflop action enum for display compatibility."""

    RAISE = auto()
    ALL_IN = auto()
    CALL = auto()
    CHECK = auto()
    FOLD = auto()


class DrawType(Enum):
    """Types of drawing hands."""

    FLUSH_DRAW = auto()          # 4 to a flush (9 outs)
    OPEN_ENDED_STRAIGHT = auto()  # OESD (8 outs)
    GUTSHOT_STRAIGHT = auto()     # Gutshot (4 outs)
    COMBO_DRAW = auto()           # Flush draw + straight draw
    NO_DRAW = auto()


@dataclass(frozen=True)
class ActionRecommendation:
    """A recommended postflop action with supporting analysis.

    Attributes:
        action: Recommended action ("raise", "call", or "fold").
        confidence: Confidence in the recommendation (0.0-1.0).
        equity: Hero's hand equity (0.0-1.0).
        pot_odds: Pot odds required to call (0.0-1.0).
        reasoning: Human-readable explanation.
        draw_type: Type of draw detected, if any.
    """

    action: str
    confidence: float
    equity: float
    pot_odds: float
    reasoning: str
    draw_type: DrawType = DrawType.NO_DRAW


# ---------------------------------------------------------------------------
# Draw detection
# ---------------------------------------------------------------------------


def detect_flush_draw(
    hand: tuple[Card, Card], board: list[Card]
) -> bool:
    """Check if hero has a flush draw (4 cards to a flush).

    Args:
        hand: Hero's two hole cards.
        board: Community cards.

    Returns:
        True if a flush draw exists using at least one hole card.
    """
    all_cards = [hand[0], hand[1]] + list(board)
    suit_counts: dict[Suit, int] = {}
    for card in all_cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

    for suit, count in suit_counts.items():
        if count == 4:
            # Ensure at least one hole card contributes to the draw.
            hero_in_suit = sum(1 for c in hand if c.suit == suit)
            if hero_in_suit >= 1:
                return True
    return False


def detect_straight_draw(
    hand: tuple[Card, Card], board: list[Card]
) -> DrawType:
    """Detect open-ended or gutshot straight draws.

    Args:
        hand: Hero's two hole cards.
        board: Community cards.

    Returns:
        DrawType.OPEN_ENDED_STRAIGHT, DrawType.GUTSHOT_STRAIGHT,
        or DrawType.NO_DRAW.
    """
    all_cards = [hand[0], hand[1]] + list(board)
    vals = sorted(set(_RANK_VAL[c.rank] for c in all_cards))

    # Also consider ace-low (ace = -1 for wheel draws).
    if 12 in vals:
        vals_with_low = sorted(set(vals + [-1]))
    else:
        vals_with_low = vals

    hero_vals = {_RANK_VAL[c.rank] for c in hand}
    if 12 in hero_vals:
        hero_vals.add(-1)

    best_draw = DrawType.NO_DRAW

    # Check all windows of 5 consecutive ranks.
    for base in range(-1, 13):
        window = set(range(base, base + 5))
        present = window & set(vals_with_low)
        missing = window - present

        if len(present) == 4 and len(missing) == 1:
            # Must involve at least one hole card.
            if hero_vals & present:
                # Determine OESD vs gutshot: OESD if the missing card
                # is at either end of the window.
                missing_val = missing.pop()
                if missing_val == base or missing_val == base + 4:
                    if best_draw != DrawType.OPEN_ENDED_STRAIGHT:
                        best_draw = DrawType.OPEN_ENDED_STRAIGHT
                else:
                    if best_draw == DrawType.NO_DRAW:
                        best_draw = DrawType.GUTSHOT_STRAIGHT

    return best_draw


def detect_draws(
    hand: tuple[Card, Card], board: list[Card]
) -> DrawType:
    """Detect the best draw type for the hero's hand.

    Args:
        hand: Hero's two hole cards.
        board: Community cards.

    Returns:
        The strongest DrawType found.
    """
    has_flush = detect_flush_draw(hand, board)
    straight_draw = detect_straight_draw(hand, board)

    if has_flush and straight_draw in (
        DrawType.OPEN_ENDED_STRAIGHT,
        DrawType.GUTSHOT_STRAIGHT,
    ):
        return DrawType.COMBO_DRAW
    if has_flush:
        return DrawType.FLUSH_DRAW
    return straight_draw


# ---------------------------------------------------------------------------
# Implied odds adjustments
# ---------------------------------------------------------------------------

# Extra equity credit for draws based on implied odds.
_IMPLIED_ODDS_BONUS: dict[DrawType, float] = {
    DrawType.COMBO_DRAW: 0.12,
    DrawType.FLUSH_DRAW: 0.08,
    DrawType.OPEN_ENDED_STRAIGHT: 0.06,
    DrawType.GUTSHOT_STRAIGHT: 0.03,
    DrawType.NO_DRAW: 0.0,
}


# ---------------------------------------------------------------------------
# PostflopAdvisor
# ---------------------------------------------------------------------------


class PostflopAdvisor:
    """Equity-based postflop decision advisor.

    Compares calculated hand equity (with implied odds adjustments for
    draws) against pot odds to recommend raise, call, or fold.
    """

    def __init__(
        self,
        equity_calculator: Optional[EquityCalculator] = None,
        num_simulations: int = 5_000,
    ) -> None:
        """Initialize the advisor.

        Args:
            equity_calculator: An EquityCalculator instance. If None,
                a new one is created.
            num_simulations: Number of Monte Carlo simulations per
                equity calculation.
        """
        self._calc = equity_calculator or EquityCalculator()
        self._num_simulations = num_simulations

    def get_recommendation(
        self,
        game_state: GameState,
        hero_hand: tuple[Card, Card],
        amount_to_call: Optional[float] = None,
    ) -> ActionRecommendation:
        """Get a postflop action recommendation.

        Args:
            game_state: Current game state with pot, board, and players.
            hero_hand: Hero's two hole cards.
            amount_to_call: Explicit bet amount to call. If None, it is
                inferred from the game state (max opponent bet minus
                hero's current bet).

        Returns:
            ActionRecommendation with action, equity, pot odds, and reasoning.
        """
        board = game_state.community_cards
        pot = game_state.get_current_pot()
        num_active = len(game_state.get_active_players())
        num_opponents = max(num_active - 1, 1)

        # Determine amount to call.
        if amount_to_call is None:
            amount_to_call = self._infer_amount_to_call(game_state)

        # Handle check opportunity (no bet to call).
        if amount_to_call <= 0:
            return self._recommend_when_checked_to(
                hero_hand, board, pot, num_opponents
            )

        # Calculate pot odds.
        pot_odds = amount_to_call / (pot + amount_to_call)

        # Calculate equity.
        equity = self._calc.calculate_equity(
            hero_hand, board, num_opponents, self._num_simulations
        )

        # Detect draws for implied odds.
        draw_type = detect_draws(hero_hand, board)
        implied_bonus = _IMPLIED_ODDS_BONUS.get(draw_type, 0.0)

        # Only apply implied odds on flop/turn (still cards to come).
        if game_state.current_street in (Street.FLOP, Street.TURN):
            adjusted_equity = equity + implied_bonus
        else:
            adjusted_equity = equity

        return self._decide(
            equity=equity,
            adjusted_equity=adjusted_equity,
            pot_odds=pot_odds,
            draw_type=draw_type,
            pot=pot,
            amount_to_call=amount_to_call,
            street=game_state.current_street,
        )

    def _infer_amount_to_call(self, game_state: GameState) -> float:
        """Infer the amount hero needs to call from game state.

        Looks at the maximum current bet among opponents minus hero's
        current bet.
        """
        hero = game_state.get_hero()
        if hero is None:
            return 0.0

        hero_bet = hero.current_bet
        max_bet = 0.0
        for player in game_state.get_active_players():
            if player.seat_number != game_state.hero_seat:
                max_bet = max(max_bet, player.current_bet)

        return max(max_bet - hero_bet, 0.0)

    def _recommend_when_checked_to(
        self,
        hero_hand: tuple[Card, Card],
        board: list[Card],
        pot: float,
        num_opponents: int,
    ) -> ActionRecommendation:
        """Recommend action when checked to hero (no bet to call).

        Uses equity thresholds to decide between betting and checking.
        """
        equity = self._calc.calculate_equity(
            hero_hand, board, num_opponents, self._num_simulations
        )

        # Evaluate current hand strength.
        all_cards = list(hero_hand) + board
        if len(all_cards) >= 5:
            strength, _ = evaluate_hand(all_cards)
        else:
            strength = HandStrength.HIGH_CARD

        draw_type = detect_draws(hero_hand, board)

        # Bet with strong hands or semi-bluff with draws.
        if equity > 0.6:
            return ActionRecommendation(
                action="raise",
                confidence=min(equity, 0.95),
                equity=equity,
                pot_odds=0.0,
                reasoning=(
                    f"Strong equity ({equity:.0%}) warrants a value bet. "
                    f"Hand strength: {strength.name}."
                ),
                draw_type=draw_type,
            )

        if equity > 0.4 and draw_type in (
            DrawType.FLUSH_DRAW,
            DrawType.OPEN_ENDED_STRAIGHT,
            DrawType.COMBO_DRAW,
        ):
            return ActionRecommendation(
                action="raise",
                confidence=0.6,
                equity=equity,
                pot_odds=0.0,
                reasoning=(
                    f"Semi-bluff opportunity with {draw_type.name} "
                    f"and equity {equity:.0%}."
                ),
                draw_type=draw_type,
            )

        # Check with medium / weak hands.
        action = "call"  # "call" here means check (no bet to call).
        return ActionRecommendation(
            action=action,
            confidence=0.5,
            equity=equity,
            pot_odds=0.0,
            reasoning=(
                f"Equity ({equity:.0%}) does not warrant betting. "
                f"Check and re-evaluate."
            ),
            draw_type=draw_type,
        )

    def _decide(
        self,
        equity: float,
        adjusted_equity: float,
        pot_odds: float,
        draw_type: DrawType,
        pot: float,
        amount_to_call: float,
        street: Street,
    ) -> ActionRecommendation:
        """Core decision logic comparing equity to pot odds.

        Thresholds:
        - adjusted_equity > pot_odds * 1.2  -> raise
        - adjusted_equity > pot_odds        -> call
        - adjusted_equity < pot_odds        -> fold (with draw nuance)
        """
        raise_threshold = pot_odds * 1.2
        call_threshold = pot_odds

        # Draw nuance: do not fold strong draws even when slightly
        # below pot odds, as implied odds justify continuing.
        is_strong_draw = draw_type in (
            DrawType.COMBO_DRAW,
            DrawType.FLUSH_DRAW,
            DrawType.OPEN_ENDED_STRAIGHT,
        )

        if adjusted_equity > raise_threshold:
            confidence = min(
                (adjusted_equity - raise_threshold) / 0.2 + 0.6, 0.95
            )
            reasoning = (
                f"Equity ({equity:.0%}) exceeds raise threshold "
                f"({raise_threshold:.0%}). Pot odds: {pot_odds:.0%}."
            )
            if draw_type != DrawType.NO_DRAW:
                reasoning += f" Draw: {draw_type.name}."
            return ActionRecommendation(
                action="raise",
                confidence=confidence,
                equity=equity,
                pot_odds=pot_odds,
                reasoning=reasoning,
                draw_type=draw_type,
            )

        if adjusted_equity > call_threshold:
            confidence = 0.5 + (adjusted_equity - call_threshold) * 2
            confidence = min(confidence, 0.85)
            reasoning = (
                f"Equity ({equity:.0%}) exceeds pot odds ({pot_odds:.0%}). "
                f"Calling is profitable."
            )
            if draw_type != DrawType.NO_DRAW:
                reasoning += (
                    f" {draw_type.name} provides additional implied odds."
                )
            return ActionRecommendation(
                action="call",
                confidence=confidence,
                equity=equity,
                pot_odds=pot_odds,
                reasoning=reasoning,
                draw_type=draw_type,
            )

        # Below pot odds: fold unless strong draw with implied odds.
        if is_strong_draw and street in (Street.FLOP, Street.TURN):
            gap = call_threshold - adjusted_equity
            if gap < 0.08:
                reasoning = (
                    f"Equity ({equity:.0%}) slightly below pot odds "
                    f"({pot_odds:.0%}), but {draw_type.name} implied odds "
                    f"justify calling."
                )
                return ActionRecommendation(
                    action="call",
                    confidence=0.4,
                    equity=equity,
                    pot_odds=pot_odds,
                    reasoning=reasoning,
                    draw_type=draw_type,
                )

        reasoning = (
            f"Equity ({equity:.0%}) below pot odds ({pot_odds:.0%}). "
            f"Fold to minimize losses."
        )
        if draw_type != DrawType.NO_DRAW:
            reasoning += f" {draw_type.name} not strong enough to continue."
        return ActionRecommendation(
            action="fold",
            confidence=min(0.5 + (pot_odds - adjusted_equity) * 2, 0.95),
            equity=equity,
            pot_odds=pot_odds,
            reasoning=reasoning,
            draw_type=draw_type,
        )
