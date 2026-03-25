"""Hand equity calculator using Monte Carlo simulation.

Estimates the probability of winning for a given hand against
opponent ranges by simulating random runouts.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from src.detection.card import Card, Rank, Suit

logger = logging.getLogger(__name__)


@dataclass
class EquityResult:
    """Result of an equity calculation.

    Attributes:
        equity: Win probability (0.0-1.0).
        simulations: Number of simulations run.
        win_count: Number of simulations won.
        tie_count: Number of simulations tied.
    """

    equity: float
    simulations: int
    win_count: int
    tie_count: int


class EquityCalculator:
    """Calculates hand equity via Monte Carlo simulation.

    Args:
        num_simulations: Default number of simulations to run.
    """

    def __init__(self, num_simulations: int = 10000) -> None:
        self._num_simulations = num_simulations

    def calculate_equity(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        num_opponents: int = 1,
        simulations: int | None = None,
    ) -> EquityResult:
        """Calculate equity for hole cards against random opponents.

        Args:
            hole_cards: Player's hole cards (2 cards).
            community_cards: Current community cards (0-5).
            num_opponents: Number of opponents to simulate.
            simulations: Override for number of simulations.

        Returns:
            EquityResult with win probability.
        """
        n = simulations or self._num_simulations

        if len(hole_cards) != 2:
            return EquityResult(equity=0.0, simulations=0, win_count=0, tie_count=0)

        # Build deck excluding known cards
        known = set()
        for c in hole_cards:
            known.add((c.rank, c.suit))
        for c in community_cards:
            known.add((c.rank, c.suit))

        deck = [
            Card(rank=r, suit=s)
            for r in Rank
            for s in Suit
            if (r, s) not in known
        ]

        wins = 0
        ties = 0
        cards_needed = 5 - len(community_cards)

        for _ in range(n):
            shuffled = random.sample(deck, cards_needed + 2 * num_opponents)

            board = list(community_cards) + shuffled[:cards_needed]
            hero_hand = list(hole_cards) + board

            # Simple hand ranking placeholder
            hero_score = self._hand_score(hero_hand)

            is_best = True
            is_tied = False
            idx = cards_needed
            for _ in range(num_opponents):
                opp_cards = shuffled[idx : idx + 2]
                opp_hand = opp_cards + board
                opp_score = self._hand_score(opp_hand)
                if opp_score > hero_score:
                    is_best = False
                    break
                elif opp_score == hero_score:
                    is_tied = True
                idx += 2

            if is_best and not is_tied:
                wins += 1
            elif is_best and is_tied:
                ties += 1

        equity = (wins + ties * 0.5) / n if n > 0 else 0.0
        return EquityResult(
            equity=equity,
            simulations=n,
            win_count=wins,
            tie_count=ties,
        )

    @staticmethod
    def _hand_score(cards: list[Card]) -> int:
        """Compute a simplified hand strength score.

        This is a placeholder; a full implementation would compute
        proper poker hand rankings.

        Args:
            cards: All 7 cards (2 hole + 5 community).

        Returns:
            An integer score (higher is better).
        """
        rank_values = {r: i for i, r in enumerate(Rank)}
        score = sum(rank_values.get(c.rank, 0) for c in cards)

        # Bonus for pairs
        rank_counts: dict[Rank, int] = {}
        for c in cards:
            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

        for rank, count in rank_counts.items():
            if count >= 2:
                score += rank_values.get(rank, 0) * count * 10

        return score
