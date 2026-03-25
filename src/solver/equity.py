"""Hand equity calculator using Monte Carlo simulation.

Evaluates poker hand strength and calculates equity against random
opponent holdings via simulation.  Optimized for speed with integer
rank representations and early-exit evaluation logic.

Usage:
    from src.solver.equity import EquityCalculator
    from src.detection.card import Card, Rank, Suit

    calc = EquityCalculator()
    equity = calc.calculate_equity(
        hand=(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)),
        board=[],
        num_opponents=1,
    )
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from src.detection.card import Card, Rank, Suit
from src.engine.game_state import HandStrength

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Integer rank values for fast comparison (2=0 .. Ace=12).
_RANK_VAL: dict[Rank, int] = {
    Rank.TWO: 0,
    Rank.THREE: 1,
    Rank.FOUR: 2,
    Rank.FIVE: 3,
    Rank.SIX: 4,
    Rank.SEVEN: 5,
    Rank.EIGHT: 6,
    Rank.NINE: 7,
    Rank.TEN: 8,
    Rank.JACK: 9,
    Rank.QUEEN: 10,
    Rank.KING: 11,
    Rank.ACE: 12,
}

# Pre-built full deck for cloning.
_FULL_DECK: list[Card] = [
    Card(rank=r, suit=s) for r in Rank for s in Suit
]

# HandStrength ordering for comparison.
_HAND_STRENGTH_VAL: dict[HandStrength, int] = {
    HandStrength.HIGH_CARD: 0,
    HandStrength.PAIR: 1,
    HandStrength.TWO_PAIR: 2,
    HandStrength.THREE_OF_A_KIND: 3,
    HandStrength.STRAIGHT: 4,
    HandStrength.FLUSH: 5,
    HandStrength.FULL_HOUSE: 6,
    HandStrength.FOUR_OF_A_KIND: 7,
    HandStrength.STRAIGHT_FLUSH: 8,
    HandStrength.ROYAL_FLUSH: 9,
}


# ---------------------------------------------------------------------------
# Hand evaluation helpers
# ---------------------------------------------------------------------------


def _rank_val(card: Card) -> int:
    """Return integer rank value for a card (2=0 .. Ace=12)."""
    return _RANK_VAL[card.rank]


def _find_straight_high(sorted_vals: list[int]) -> int | None:
    """Find the highest straight in a sorted (desc) list of unique rank values.

    Args:
        sorted_vals: Unique rank values sorted descending.

    Returns:
        The high card value of the best straight, or None.
    """
    # Add low-ace (value -1 mapped to 3 for wheel: A-2-3-4-5).
    vals = list(sorted_vals)
    if 12 in vals:
        vals.append(-1)  # Ace as low

    for i in range(len(vals) - 4):
        if vals[i] - vals[i + 4] == 4:
            # Check that all five values are consecutive.
            window = vals[i : i + 5]
            if all(window[j] - window[j + 1] == 1 for j in range(4)):
                return vals[i]
    return None


def _evaluate_five(cards: list[Card]) -> tuple[HandStrength, list[int]]:
    """Evaluate exactly 5 cards and return (strength, kickers).

    Kickers are rank values sorted for tie-breaking (highest first).

    Args:
        cards: Exactly 5 Card objects.

    Returns:
        Tuple of (HandStrength, list of kicker values).
    """
    vals = sorted((_rank_val(c) for c in cards), reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    unique_vals = sorted(set(vals), reverse=True)
    straight_high = _find_straight_high(unique_vals)
    is_straight = straight_high is not None

    counts = Counter(vals)
    freq = sorted(counts.values(), reverse=True)

    # Straight flush / Royal flush
    if is_flush and is_straight:
        if straight_high == 12:
            return (HandStrength.ROYAL_FLUSH, [12])
        return (HandStrength.STRAIGHT_FLUSH, [straight_high])

    # Four of a kind
    if freq == [4, 1]:
        quad_val = [v for v, c in counts.items() if c == 4][0]
        kicker = [v for v, c in counts.items() if c == 1][0]
        return (HandStrength.FOUR_OF_A_KIND, [quad_val, kicker])

    # Full house
    if freq == [3, 2]:
        trip_val = [v for v, c in counts.items() if c == 3][0]
        pair_val = [v for v, c in counts.items() if c == 2][0]
        return (HandStrength.FULL_HOUSE, [trip_val, pair_val])

    # Flush
    if is_flush:
        return (HandStrength.FLUSH, vals)

    # Straight
    if is_straight:
        return (HandStrength.STRAIGHT, [straight_high])

    # Three of a kind
    if freq == [3, 1, 1]:
        trip_val = [v for v, c in counts.items() if c == 3][0]
        kickers = sorted(
            [v for v, c in counts.items() if c == 1], reverse=True
        )
        return (HandStrength.THREE_OF_A_KIND, [trip_val] + kickers)

    # Two pair
    if freq == [2, 2, 1]:
        pairs = sorted(
            [v for v, c in counts.items() if c == 2], reverse=True
        )
        kicker = [v for v, c in counts.items() if c == 1][0]
        return (HandStrength.TWO_PAIR, pairs + [kicker])

    # One pair
    if freq == [2, 1, 1, 1]:
        pair_val = [v for v, c in counts.items() if c == 2][0]
        kickers = sorted(
            [v for v, c in counts.items() if c == 1], reverse=True
        )
        return (HandStrength.PAIR, [pair_val] + kickers)

    # High card
    return (HandStrength.HIGH_CARD, vals)


def _hand_key(strength: HandStrength, kickers: list[int]) -> tuple[int, ...]:
    """Convert (strength, kickers) to a sortable tuple."""
    return (_HAND_STRENGTH_VAL[strength], *kickers)


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def evaluate_hand(cards: list[Card]) -> tuple[HandStrength, list[int]]:
    """Evaluate the best 5-card hand from up to 7 cards.

    Tries all C(n, 5) combinations and returns the best.

    Args:
        cards: 5 to 7 Card objects.

    Returns:
        Tuple of (HandStrength, kickers) for the best 5-card hand.

    Raises:
        ValueError: If fewer than 5 cards are provided.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")

    if len(cards) == 5:
        return _evaluate_five(cards)

    best_key: tuple[int, ...] = (-1,)
    best_result: tuple[HandStrength, list[int]] = (
        HandStrength.HIGH_CARD,
        [],
    )

    for combo in combinations(cards, 5):
        strength, kickers = _evaluate_five(list(combo))
        key = _hand_key(strength, kickers)
        if key > best_key:
            best_key = key
            best_result = (strength, kickers)

    return best_result


def compare_hands(
    hand1_cards: list[Card],
    hand2_cards: list[Card],
    board: list[Card],
) -> int:
    """Compare two hands given a board.

    Args:
        hand1_cards: First player's hole cards (2 cards).
        hand2_cards: Second player's hole cards (2 cards).
        board: Community cards (3-5 cards).

    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 for a tie.
    """
    eval1 = evaluate_hand(list(hand1_cards) + list(board))
    eval2 = evaluate_hand(list(hand2_cards) + list(board))

    key1 = _hand_key(*eval1)
    key2 = _hand_key(*eval2)

    if key1 > key2:
        return 1
    if key1 < key2:
        return -1
    return 0


# ---------------------------------------------------------------------------
# EquityCalculator
# ---------------------------------------------------------------------------


@dataclass
class EquityResult:
    """Result of an equity calculation.

    Attributes:
        equity: Win probability as a float between 0.0 and 1.0.
        win_count: Number of simulations won.
        tie_count: Number of simulations tied.
        total_simulations: Number of simulations run.
    """

    equity: float
    win_count: int
    tie_count: int
    total_simulations: int


class EquityCalculator:
    """Monte Carlo hand equity calculator.

    Estimates the probability of winning a hand by dealing out random
    remaining cards many times and evaluating the results.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the calculator.

        Args:
            seed: Optional RNG seed for reproducible results.
        """
        self._rng = random.Random(seed)

    def calculate_equity(
        self,
        hand: tuple[Card, Card],
        board: list[Card],
        num_opponents: int = 1,
        num_simulations: int = 10_000,
    ) -> float:
        """Calculate equity of a hand vs. random opponent holdings.

        Args:
            hand: Hero's two hole cards.
            board: Community cards dealt so far (0-5).
            num_opponents: Number of opponents still in the hand.
            num_simulations: Number of Monte Carlo iterations.

        Returns:
            Equity as a float between 0.0 and 1.0.

        Raises:
            ValueError: If board has more than 5 cards or hand is invalid.
        """
        result = self.calculate_equity_detailed(
            hand, board, num_opponents, num_simulations
        )
        return result.equity

    def calculate_equity_detailed(
        self,
        hand: tuple[Card, Card],
        board: list[Card],
        num_opponents: int = 1,
        num_simulations: int = 10_000,
    ) -> EquityResult:
        """Calculate detailed equity results.

        Args:
            hand: Hero's two hole cards.
            board: Community cards dealt so far (0-5).
            num_opponents: Number of opponents still in the hand.
            num_simulations: Number of Monte Carlo iterations.

        Returns:
            EquityResult with win/tie/total counts and equity.

        Raises:
            ValueError: If board has more than 5 cards.
        """
        if len(board) > 5:
            raise ValueError(f"Board cannot have more than 5 cards, got {len(board)}")
        if num_opponents < 1:
            raise ValueError("Must have at least 1 opponent")

        # Build deck minus known cards.
        known: set[Card] = {hand[0], hand[1]} | set(board)
        deck = [c for c in _FULL_DECK if c not in known]

        cards_needed_board = 5 - len(board)
        cards_needed_opponents = 2 * num_opponents
        cards_per_sim = cards_needed_board + cards_needed_opponents

        if cards_per_sim > len(deck):
            raise ValueError(
                f"Not enough cards in deck ({len(deck)}) for simulation "
                f"needing {cards_per_sim} cards"
            )

        wins = 0
        ties = 0

        hero_cards = list(hand)

        for _ in range(num_simulations):
            # Shuffle and deal.
            drawn = self._rng.sample(deck, cards_per_sim)

            sim_board = list(board) + drawn[:cards_needed_board]
            hero_all = hero_cards + sim_board
            hero_eval = _hand_key(*evaluate_hand(hero_all))

            # Evaluate opponents.
            hero_wins = True
            any_tie = False
            opp_start = cards_needed_board

            for opp_idx in range(num_opponents):
                opp_cards_start = opp_start + opp_idx * 2
                opp_hole = drawn[opp_cards_start : opp_cards_start + 2]
                opp_all = opp_hole + sim_board
                opp_eval = _hand_key(*evaluate_hand(opp_all))

                if opp_eval > hero_eval:
                    hero_wins = False
                    any_tie = False
                    break
                elif opp_eval == hero_eval:
                    any_tie = True

            if hero_wins:
                if any_tie:
                    ties += 1
                else:
                    wins += 1

        # Equity = wins + ties/2 (split pot approximation).
        equity = (wins + ties * 0.5) / num_simulations

        return EquityResult(
            equity=equity,
            win_count=wins,
            tie_count=ties,
            total_simulations=num_simulations,
        )
