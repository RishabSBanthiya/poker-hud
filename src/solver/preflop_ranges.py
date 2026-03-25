"""Preflop GTO range lookup tables for the solver subsystem.

Provides hand range representations and position-based preflop strategy
tables derived from simplified GTO solutions.  Hands are represented in
standard notation: "AKs" (suited), "AKo" (offsuit), "AA" (pocket pair).

Usage:
    from src.solver.preflop_ranges import PreflopRangeTable
    from src.engine.game_state import Position

    table = PreflopRangeTable()
    open_range = table.get_open_range(Position.BTN)
    rec = table.get_recommendation(hand, Position.BTN, action_facing="open")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.detection.card import Card, Rank
from src.engine.game_state import Position

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rank ordering for matrix construction (Ace highest).
RANK_ORDER: list[Rank] = [
    Rank.ACE,
    Rank.KING,
    Rank.QUEEN,
    Rank.JACK,
    Rank.TEN,
    Rank.NINE,
    Rank.EIGHT,
    Rank.SEVEN,
    Rank.SIX,
    Rank.FIVE,
    Rank.FOUR,
    Rank.THREE,
    Rank.TWO,
]

_RANK_INDEX: dict[Rank, int] = {r: i for i, r in enumerate(RANK_ORDER)}

# Standard single-character notation for ranks (Ten = "T").
_RANK_CHAR: dict[Rank, str] = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.ACE: "A",
}

# Reverse mapping: char -> Rank.
_CHAR_RANK: dict[str, Rank] = {v: k for k, v in _RANK_CHAR.items()}


def rank_index(rank: Rank) -> int:
    """Return the 0-based index of a rank in the 13x13 matrix (Ace=0)."""
    return _RANK_INDEX[rank]


# ---------------------------------------------------------------------------
# Hand notation helpers
# ---------------------------------------------------------------------------


def hand_to_notation(card1: Card, card2: Card) -> str:
    """Convert two cards into standard hand notation.

    Args:
        card1: First hole card.
        card2: Second hole card.

    Returns:
        A string like "AKs", "AKo", or "AA".
    """
    r1_idx = rank_index(card1.rank)
    r2_idx = rank_index(card2.rank)

    # Ensure higher rank comes first in notation.
    if r1_idx > r2_idx:
        card1, card2 = card2, card1

    high = _RANK_CHAR[card1.rank]
    low = _RANK_CHAR[card2.rank]

    if card1.rank == card2.rank:
        return f"{high}{low}"
    elif card1.suit == card2.suit:
        return f"{high}{low}s"
    else:
        return f"{high}{low}o"


def notation_to_matrix_pos(notation: str) -> tuple[int, int]:
    """Convert hand notation to (row, col) in the 13x13 matrix.

    In the standard matrix:
    - Pocket pairs are on the diagonal.
    - Suited hands are above the diagonal (row < col).
    - Offsuit hands are below the diagonal (row > col).

    Args:
        notation: Hand string like "AKs", "AKo", "AA".

    Returns:
        (row, col) tuple for the 13x13 matrix.
    """
    high_char = notation[0]
    low_char = notation[1]

    high_rank = _CHAR_RANK[high_char]
    low_rank = _CHAR_RANK[low_char]

    high_idx = rank_index(high_rank)
    low_idx = rank_index(low_rank)

    if len(notation) == 2:
        # Pocket pair
        return (high_idx, high_idx)
    elif notation[2] == "s":
        # Suited: above diagonal
        return (high_idx, low_idx)
    else:
        # Offsuit: below diagonal
        return (low_idx, high_idx)


# ---------------------------------------------------------------------------
# HandRange
# ---------------------------------------------------------------------------


@dataclass
class HandRange:
    """A set of hands with weights for mixed strategies.

    Each hand is stored as a notation string (e.g. "AKs") mapped to a
    weight between 0.0 and 1.0, where 1.0 means always play and 0.5
    means play 50% of the time.

    Attributes:
        hands: Mapping of hand notation to weight.
    """

    hands: dict[str, float] = field(default_factory=dict)

    def add_hand(self, notation: str, weight: float = 1.0) -> None:
        """Add a hand to the range with a given weight.

        Args:
            notation: Hand notation (e.g. "AKs").
            weight: Frequency weight between 0.0 and 1.0.

        Raises:
            ValueError: If weight is outside [0.0, 1.0].
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be 0.0-1.0, got {weight}")
        self.hands[notation] = weight

    def remove_hand(self, notation: str) -> None:
        """Remove a hand from the range.

        Args:
            notation: Hand notation to remove.
        """
        self.hands.pop(notation, None)

    def contains(self, notation: str) -> bool:
        """Check whether a hand is in the range (weight > 0).

        Args:
            notation: Hand notation to check.

        Returns:
            True if the hand is in the range with weight > 0.
        """
        return self.hands.get(notation, 0.0) > 0.0

    def get_weight(self, notation: str) -> float:
        """Return the weight for a hand, or 0.0 if not in range.

        Args:
            notation: Hand notation to look up.

        Returns:
            Weight between 0.0 and 1.0.
        """
        return self.hands.get(notation, 0.0)

    @property
    def size(self) -> int:
        """Number of hands in the range with weight > 0."""
        return sum(1 for w in self.hands.values() if w > 0.0)

    @property
    def total_combos(self) -> float:
        """Approximate total combo count, weighted.

        Pocket pairs have 6 combos, suited hands 4, offsuit hands 12.
        """
        total = 0.0
        for notation, weight in self.hands.items():
            if weight <= 0.0:
                continue
            if len(notation) == 2:
                total += 6 * weight
            elif notation[2] == "s":
                total += 4 * weight
            else:
                total += 12 * weight
        return total

    def __len__(self) -> int:
        return self.size

    def __contains__(self, notation: str) -> bool:
        return self.contains(notation)


# ---------------------------------------------------------------------------
# Preflop range data
# ---------------------------------------------------------------------------

def _build_range(hand_list: list[str | tuple[str, float]]) -> HandRange:
    """Build a HandRange from a list of notations or (notation, weight) tuples."""
    hr = HandRange()
    for item in hand_list:
        if isinstance(item, tuple):
            hr.add_hand(item[0], item[1])
        else:
            hr.add_hand(item, 1.0)
    return hr


# --- Open raise ranges by position (6-max simplified GTO) ---

_UTG_OPEN: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ", "TT", "99",
    ("88", 0.5),
    "AKs", "AQs", "AJs", "ATs",
    "KQs", "KJs",
    ("KTs", 0.5),
    "QJs", "QTs",
    "JTs",
    "AKo", "AQo",
    ("AJo", 0.5),
]

_MP_OPEN: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ", "TT", "99", "88",
    ("77", 0.5),
    "AKs", "AQs", "AJs", "ATs", "A9s",
    "KQs", "KJs", "KTs",
    "QJs", "QTs",
    "JTs", "J9s",
    "T9s",
    "AKo", "AQo", "AJo",
    ("ATo", 0.5),
    "KQo",
]

_CO_OPEN: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
    ("55", 0.5),
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s",
    "KQs", "KJs", "KTs", "K9s",
    "QJs", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "T8s",
    "98s",
    ("87s", 0.5),
    "AKo", "AQo", "AJo", "ATo",
    "KQo", "KJo",
    ("KTo", 0.5),
    "QJo",
    ("QTo", 0.5),
]

_BTN_OPEN: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44",
    ("33", 0.5), ("22", 0.5),
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s",
    "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s",
    ("K5s", 0.5),
    "QJs", "QTs", "Q9s", "Q8s",
    "JTs", "J9s", "J8s",
    "T9s", "T8s",
    "98s", "97s",
    "87s", "86s",
    "76s", "75s",
    "65s",
    ("54s", 0.5),
    "AKo", "AQo", "AJo", "ATo", "A9o",
    ("A8o", 0.5),
    "KQo", "KJo", "KTo",
    ("K9o", 0.5),
    "QJo", "QTo",
    ("Q9o", 0.5),
    "JTo",
    ("J9o", 0.5),
    "T9o",
]

_SB_OPEN: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
    ("44", 0.5), ("33", 0.5),
    "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s",
    "A4s", "A3s", "A2s",
    "KQs", "KJs", "KTs", "K9s", "K8s", "K7s",
    "QJs", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "T8s",
    "98s", "97s",
    "87s", "86s",
    "76s",
    "65s",
    "AKo", "AQo", "AJo", "ATo", "A9o",
    "KQo", "KJo", "KTo",
    "QJo", "QTo",
    "JTo",
]

_OPEN_RANGES: dict[Position, list[str | tuple[str, float]]] = {
    Position.UTG: _UTG_OPEN,
    Position.UTG1: _UTG_OPEN,
    Position.LJ: _MP_OPEN,
    Position.MP: _MP_OPEN,
    Position.MP1: _MP_OPEN,
    Position.HJ: _CO_OPEN,
    Position.CO: _CO_OPEN,
    Position.BTN: _BTN_OPEN,
    Position.SB: _SB_OPEN,
}

# --- 3-bet ranges (simplified: tighter in EP vs EP, wider in LP vs EP) ---

_TIGHT_3BET: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ",
    ("JJ", 0.5),
    "AKs",
    ("AKo", 0.5),
]

_MEDIUM_3BET: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ",
    ("TT", 0.5),
    "AKs", "AQs",
    ("AJs", 0.5),
    "AKo",
    ("AQo", 0.5),
]

_WIDE_3BET: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ", "JJ", "TT",
    ("99", 0.5),
    "AKs", "AQs", "AJs", "ATs",
    ("A5s", 0.5),  # Suited ace blocker
    "KQs",
    ("KJs", 0.5),
    "AKo", "AQo",
    ("AJo", 0.5),
]

# --- Call vs open ranges ---

_TIGHT_CALL: list[str | tuple[str, float]] = [
    "JJ", "TT", "99", "88", "77",
    "AQs", "AJs", "ATs",
    "KQs", "KJs",
    "QJs",
    "JTs",
    "T9s",
    "98s",
]

_MEDIUM_CALL: list[str | tuple[str, float]] = [
    "TT", "99", "88", "77", "66",
    ("55", 0.5),
    "AJs", "ATs", "A9s", "A8s",
    "KQs", "KJs", "KTs",
    "QJs", "QTs",
    "JTs", "J9s",
    "T9s",
    "98s",
    "87s",
    "76s",
    "AQo", "AJo",
    "KQo",
]

_WIDE_CALL: list[str | tuple[str, float]] = [
    "TT", "99", "88", "77", "66", "55", "44",
    ("33", 0.5), ("22", 0.5),
    "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s",
    "KQs", "KJs", "KTs", "K9s",
    "QJs", "QTs", "Q9s",
    "JTs", "J9s",
    "T9s", "T8s",
    "98s", "97s",
    "87s", "86s",
    "76s", "75s",
    "65s",
    "54s",
    "AQo", "AJo", "ATo",
    "KQo", "KJo",
    "QJo",
    "JTo",
]

# --- 4-bet ranges (simplified) ---

_FOURBET: list[str | tuple[str, float]] = [
    "AA", "KK", "QQ",
    ("JJ", 0.5),
    "AKs",
    ("AKo", 0.5),
]


# ---------------------------------------------------------------------------
# PreflopRangeTable
# ---------------------------------------------------------------------------


_EP_POSITIONS = {Position.UTG, Position.UTG1, Position.MP, Position.MP1, Position.LJ}
_LP_POSITIONS = {Position.HJ, Position.CO, Position.BTN, Position.SB}


class PreflopRangeTable:
    """Preflop GTO range lookup table.

    Provides position-based ranges for open raising, 3-betting, calling,
    and 4-betting.  Ranges are approximate simplified GTO solutions for
    6-max cash games.
    """

    def __init__(self) -> None:
        self._open_cache: dict[Position, HandRange] = {}
        self._three_bet_cache: dict[tuple[Position, Position], HandRange] = {}
        self._call_cache: dict[tuple[Position, Position], HandRange] = {}
        self._four_bet_range: Optional[HandRange] = None

    def get_open_range(self, position: Position) -> HandRange:
        """Return the open raise range for a position.

        Args:
            position: The player's table position.

        Returns:
            HandRange containing open-raise hands and weights.
        """
        if position in self._open_cache:
            return self._open_cache[position]

        range_data = _OPEN_RANGES.get(position, _UTG_OPEN)
        hr = _build_range(range_data)
        self._open_cache[position] = hr
        return hr

    def get_three_bet_range(
        self, position: Position, vs_position: Position
    ) -> HandRange:
        """Return the 3-bet range for *position* facing an open from *vs_position*.

        Uses a simplified model:
        - LP vs EP: wide 3-bet range
        - LP vs LP: medium 3-bet range
        - EP vs EP: tight 3-bet range

        Args:
            position: The 3-bettor's position.
            vs_position: The original opener's position.

        Returns:
            HandRange for 3-betting.
        """
        key = (position, vs_position)
        if key in self._three_bet_cache:
            return self._three_bet_cache[key]

        if position in _LP_POSITIONS and vs_position in _EP_POSITIONS:
            data = _WIDE_3BET
        elif position in _LP_POSITIONS and vs_position in _LP_POSITIONS:
            data = _MEDIUM_3BET
        else:
            data = _TIGHT_3BET

        hr = _build_range(data)
        self._three_bet_cache[key] = hr
        return hr

    def get_call_range(
        self, position: Position, vs_position: Position
    ) -> HandRange:
        """Return the calling range for *position* facing an open from *vs_position*.

        Uses a simplified model:
        - BB vs any: wide calling range
        - LP vs EP: medium calling range
        - EP vs EP: tight calling range

        Args:
            position: The caller's position.
            vs_position: The opener's position.

        Returns:
            HandRange for calling.
        """
        key = (position, vs_position)
        if key in self._call_cache:
            return self._call_cache[key]

        if position == Position.BB:
            data = _WIDE_CALL
        elif position in _LP_POSITIONS:
            data = _MEDIUM_CALL
        else:
            data = _TIGHT_CALL

        hr = _build_range(data)
        self._call_cache[key] = hr
        return hr

    def get_four_bet_range(self) -> HandRange:
        """Return the simplified 4-bet range (position-independent).

        Returns:
            HandRange for 4-betting.
        """
        if self._four_bet_range is not None:
            return self._four_bet_range

        self._four_bet_range = _build_range(_FOURBET)
        return self._four_bet_range

    def is_in_range(self, hand: tuple[Card, Card], hr: HandRange) -> bool:
        """Check whether a specific hand is in a given range.

        Args:
            hand: Tuple of two hole cards.
            hr: The HandRange to check against.

        Returns:
            True if the hand notation is contained in the range.
        """
        notation = hand_to_notation(hand[0], hand[1])
        return hr.contains(notation)

    def get_recommendation(
        self,
        hand: tuple[Card, Card],
        position: Position,
        action_facing: str = "open",
        vs_position: Optional[Position] = None,
    ) -> str:
        """Get a preflop action recommendation for a hand.

        Args:
            hand: Tuple of two hole cards.
            position: Hero's table position.
            action_facing: One of "open", "raise" (facing an open),
                "three_bet" (facing a 3-bet).
            vs_position: Opponent's position when facing action.
                Required for "raise" and "three_bet" actions.

        Returns:
            Recommendation string: "raise", "call", or "fold".
        """
        notation = hand_to_notation(hand[0], hand[1])

        if action_facing == "open":
            open_range = self.get_open_range(position)
            if open_range.contains(notation):
                return "raise"
            return "fold"

        if action_facing == "raise":
            opener = vs_position or Position.UTG
            three_bet_range = self.get_three_bet_range(position, opener)
            if three_bet_range.contains(notation):
                return "raise"
            call_range = self.get_call_range(position, opener)
            if call_range.contains(notation):
                return "call"
            return "fold"

        if action_facing == "three_bet":
            four_bet_range = self.get_four_bet_range()
            if four_bet_range.contains(notation):
                return "raise"
            # Simplified: call with premium hands not in 4-bet range
            opener = vs_position or Position.UTG
            call_range = self.get_call_range(position, opener)
            if call_range.contains(notation):
                return "call"
            return "fold"

        logger.warning("Unknown action_facing=%r, defaulting to fold", action_facing)
        return "fold"
