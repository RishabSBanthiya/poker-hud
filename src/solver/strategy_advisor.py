"""Strategy advisor providing GTO-based recommendations.

Combines equity calculations with game state to produce
actionable advice for the player.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from src.engine.game_state import HandState
from src.solver.equity_calculator import EquityCalculator, EquityResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyAdvice:
    """Strategy recommendation for the current game state.

    Attributes:
        action: Recommended action (fold, call, raise).
        confidence: Confidence in the recommendation (0.0-1.0).
        equity: Current hand equity result.
        explanation: Human-readable explanation.
        pot_odds: Current pot odds percentage.
    """

    action: str
    confidence: float
    equity: EquityResult
    explanation: str
    pot_odds: float = 0.0


class StrategyAdvisorCoordinator:
    """Produces strategy recommendations from game state.

    Analyzes the current hand state, calculates equity,
    and recommends actions based on pot odds vs equity.

    Args:
        equity_calculator: Calculator for hand equity.
    """

    def __init__(self, equity_calculator: EquityCalculator) -> None:
        self._equity_calc = equity_calculator
        self._on_advice: Optional[
            Callable[[StrategyAdvice], None]
        ] = None

    def set_advice_callback(
        self, callback: Callable[[StrategyAdvice], None]
    ) -> None:
        """Set callback for new strategy advice.

        Args:
            callback: Function receiving StrategyAdvice.
        """
        self._on_advice = callback

    def analyze_state(self, hand_state: HandState) -> Optional[StrategyAdvice]:
        """Analyze game state and produce strategy advice.

        Args:
            hand_state: Current hand state.

        Returns:
            StrategyAdvice, or None if insufficient information.
        """
        hero = next(
            (p for p in hand_state.players if p.is_hero), None
        )
        if hero is None or len(hero.hole_cards) != 2:
            return None

        num_opponents = len(hand_state.get_active_players()) - 1
        if num_opponents < 1:
            num_opponents = 1

        equity_result = self._equity_calc.calculate_equity(
            hole_cards=hero.hole_cards,
            community_cards=hand_state.community_cards,
            num_opponents=num_opponents,
            simulations=5000,
        )

        pot_odds = self._calculate_pot_odds(hand_state)
        advice = self._recommend_action(equity_result, pot_odds, hand_state)

        if self._on_advice is not None:
            self._on_advice(advice)

        return advice

    @staticmethod
    def _calculate_pot_odds(hand_state: HandState) -> float:
        """Calculate pot odds as a percentage.

        Args:
            hand_state: Current hand state.

        Returns:
            Pot odds percentage (0.0-100.0).
        """
        if hand_state.pot <= 0:
            return 0.0
        bet_to_call = hand_state.big_blind  # Simplified
        return (bet_to_call / (hand_state.pot + bet_to_call)) * 100

    @staticmethod
    def _recommend_action(
        equity: EquityResult,
        pot_odds: float,
        hand_state: HandState,
    ) -> StrategyAdvice:
        """Determine the recommended action.

        Args:
            equity: Calculated hand equity.
            pot_odds: Current pot odds percentage.
            hand_state: Current hand state.

        Returns:
            StrategyAdvice with the recommendation.
        """
        equity_pct = equity.equity * 100

        if equity_pct > 70:
            action = "raise"
            explanation = (
                f"Strong hand ({equity_pct:.0f}% equity). Raise for value."
            )
            confidence = min(equity.equity, 0.95)
        elif equity_pct > pot_odds:
            action = "call"
            explanation = (
                f"Equity ({equity_pct:.0f}%) exceeds pot odds ({pot_odds:.0f}%). "
                f"Profitable call."
            )
            confidence = min((equity_pct - pot_odds) / 50, 0.8)
        else:
            action = "fold"
            explanation = (
                f"Equity ({equity_pct:.0f}%) below pot odds ({pot_odds:.0f}%). "
                f"Fold is correct."
            )
            confidence = min((pot_odds - equity_pct) / 50, 0.8)

        return StrategyAdvice(
            action=action,
            confidence=confidence,
            equity=equity,
            explanation=explanation,
            pot_odds=pot_odds,
        )
