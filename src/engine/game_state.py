"""Core game state data model for the poker engine.

Defines enums and dataclasses that represent the full state of a poker
hand in progress: positions, streets, actions, players, and the overall
game state.  Reuses Card/Suit/Rank from the detection subsystem so there
is a single canonical card representation across the codebase.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.detection.card import Card

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Position(Enum):
    """Seat positions at a poker table (supports 2-10 players)."""

    BTN = "BTN"
    SB = "SB"
    BB = "BB"
    UTG = "UTG"
    UTG1 = "UTG1"
    LJ = "LJ"
    HJ = "HJ"
    CO = "CO"
    MP = "MP"
    MP1 = "MP1"


# Canonical position orders used when assigning seats.
# Full-ring (9-10 players) order; shorter tables use a suffix slice.
POSITION_ORDER_FULL: list[Position] = [
    Position.SB,
    Position.BB,
    Position.UTG,
    Position.UTG1,
    Position.MP,
    Position.MP1,
    Position.LJ,
    Position.HJ,
    Position.CO,
    Position.BTN,
]


class Street(Enum):
    """Betting streets in a hand of Texas Hold'em."""

    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()


# Defines the expected number of community cards per street.
STREET_COMMUNITY_CARDS: dict[Street, int] = {
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
    Street.SHOWDOWN: 5,
}

# Valid transitions: current street -> next street.
_STREET_NEXT: dict[Street, Street] = {
    Street.PREFLOP: Street.FLOP,
    Street.FLOP: Street.TURN,
    Street.TURN: Street.RIVER,
    Street.RIVER: Street.SHOWDOWN,
}


class ActionType(Enum):
    """Possible player actions during a poker hand."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_BLIND = "post_blind"


class HandStrength(Enum):
    """Standard poker hand rankings, lowest to highest."""

    HIGH_CARD = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    STRAIGHT_FLUSH = auto()
    ROYAL_FLUSH = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PlayerAction:
    """A single action taken by a player.

    Attributes:
        action_type: The kind of action (fold, call, etc.).
        amount: Chip amount associated with the action.  Zero for folds /
            checks.
        street: The street on which the action occurred.
        timestamp: Unix timestamp when the action was recorded.
        player_name: Name of the player who performed the action.
    """

    action_type: ActionType
    amount: float
    street: Street
    timestamp: float = field(default_factory=time.time)
    player_name: str = ""

    def __str__(self) -> str:
        if self.amount > 0:
            return f"{self.player_name} {self.action_type.value} ${self.amount:.2f}"
        return f"{self.player_name} {self.action_type.value}"


@dataclass
class Player:
    """State of a single player at the table.

    Attributes:
        name: Display name / screen name of the player.
        seat_number: Physical seat index (0-based).
        position: Table position (BTN, SB, BB, etc.).
        stack_size: Current chip stack.
        hole_cards: The player's private cards, if known.
        is_active: Whether the player is still in the current hand
            (has not folded).
        is_all_in: Whether the player is all-in.
        current_bet: Amount the player has wagered in the current
            betting round.
        actions: Ordered list of actions the player has taken this hand.
    """

    name: str
    seat_number: int
    position: Optional[Position] = None
    stack_size: float = 0.0
    hole_cards: Optional[list[Card]] = None
    is_active: bool = True
    is_all_in: bool = False
    current_bet: float = 0.0
    actions: list[PlayerAction] = field(default_factory=list)

    def reset_for_new_street(self) -> None:
        """Reset per-street betting state (current bet) for a new street."""
        self.current_bet = 0.0

    def add_action(self, action: PlayerAction) -> None:
        """Record an action and update player state accordingly.

        Args:
            action: The action to record.
        """
        self.actions.append(action)
        if action.action_type == ActionType.FOLD:
            self.is_active = False
        elif action.action_type == ActionType.ALL_IN:
            self.is_all_in = True
            self.current_bet += action.amount
            self.stack_size = 0.0
        elif action.action_type in (
            ActionType.BET,
            ActionType.RAISE,
            ActionType.CALL,
            ActionType.POST_BLIND,
        ):
            self.current_bet += action.amount
            self.stack_size -= action.amount

    def __str__(self) -> str:
        pos = self.position.value if self.position else "?"
        return f"[Seat {self.seat_number}] {self.name} ({pos}) ${self.stack_size:.2f}"


@dataclass
class SidePot:
    """A side pot created when one or more players are all-in.

    Attributes:
        amount: Total chips in this side pot.
        eligible_players: Seat numbers of players eligible to win.
    """

    amount: float
    eligible_players: list[int] = field(default_factory=list)


@dataclass
class GameState:
    """Complete snapshot of a poker hand in progress.

    This is the central data structure shared across the engine, stats,
    solver, and overlay subsystems.

    Attributes:
        table_id: Unique identifier for the table.
        table_name: Human-readable table name.
        small_blind: Small blind amount.
        big_blind: Big blind amount.
        ante: Per-player ante (0 if no ante).
        players: All players seated at the table.
        hero_seat: Seat number of the hero (the user).
        hand_id: Unique identifier for the current hand.
        hand_number: Sequential hand number for this session.
        current_street: The current betting street.
        community_cards: Board cards dealt so far.
        pot_size: Total chips in the main pot.
        side_pots: Side pots, if any.
        action_history: Chronological list of all actions this hand.
        timestamp: Unix timestamp when this state was captured.
    """

    # Table info
    table_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    table_name: str = ""
    small_blind: float = 0.0
    big_blind: float = 0.0
    ante: float = 0.0

    # Players
    players: list[Player] = field(default_factory=list)
    hero_seat: int = 0

    # Hand state
    hand_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hand_number: int = 0
    current_street: Street = Street.PREFLOP
    community_cards: list[Card] = field(default_factory=list)
    pot_size: float = 0.0
    side_pots: list[SidePot] = field(default_factory=list)

    # History
    action_history: list[PlayerAction] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_hero(self) -> Optional[Player]:
        """Return the hero (user) player, or None if not found.

        Returns:
            The Player whose seat matches ``hero_seat``, or None.
        """
        return self.get_player_by_seat(self.hero_seat)

    def get_player_by_seat(self, seat: int) -> Optional[Player]:
        """Return the player at a given seat number.

        Args:
            seat: The seat number to look up.

        Returns:
            The matching Player, or None if no player occupies that seat.
        """
        for player in self.players:
            if player.seat_number == seat:
                return player
        return None

    def get_player_by_name(self, name: str) -> Optional[Player]:
        """Return the player with the given name.

        Args:
            name: The player name to search for.

        Returns:
            The matching Player, or None if not found.
        """
        for player in self.players:
            if player.name == name:
                return player
        return None

    def get_active_players(self) -> list[Player]:
        """Return all players still in the hand (have not folded).

        Returns:
            List of active players, ordered by seat number.
        """
        return [p for p in self.players if p.is_active]

    def get_current_pot(self) -> float:
        """Return the total pot including all side pots.

        Returns:
            Sum of the main pot and all side pots.
        """
        side_total = sum(sp.amount for sp in self.side_pots)
        return self.pot_size + side_total

    def get_num_players(self) -> int:
        """Return the total number of seated players.

        Returns:
            Count of all players at the table.
        """
        return len(self.players)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def advance_street(self) -> None:
        """Advance to the next betting street.

        Resets per-street player bets and moves ``current_street`` forward.

        Raises:
            ValueError: If the hand is already at SHOWDOWN and cannot
                advance further.
        """
        next_street = _STREET_NEXT.get(self.current_street)
        if next_street is None:
            raise ValueError(
                f"Cannot advance past {self.current_street.name}"
            )
        self.current_street = next_street
        for player in self.players:
            player.reset_for_new_street()

    def add_community_cards(self, cards: list[Card]) -> None:
        """Add community cards to the board.

        Validates that the resulting board size does not exceed 5 and that
        the number of cards is appropriate for the current street.

        Args:
            cards: Cards to add to the community board.

        Raises:
            ValueError: If adding the cards would exceed 5 total community
                cards, or if the count is invalid for the current street.
        """
        if not cards:
            return

        new_total = len(self.community_cards) + len(cards)
        if new_total > 5:
            raise ValueError(
                f"Cannot add {len(cards)} card(s): board would have "
                f"{new_total} cards (max 5)"
            )

        expected = STREET_COMMUNITY_CARDS.get(self.current_street, 0)
        if new_total > expected:
            raise ValueError(
                f"Adding {len(cards)} card(s) would give {new_total} "
                f"community cards, but {self.current_street.name} expects "
                f"at most {expected}"
            )

        self.community_cards.extend(cards)

    def record_action(self, seat: int, action: PlayerAction) -> None:
        """Record a player action, updating both player and game state.

        Args:
            seat: Seat number of the acting player.
            action: The action being performed.

        Raises:
            ValueError: If no player is found at the given seat.
        """
        player = self.get_player_by_seat(seat)
        if player is None:
            raise ValueError(f"No player at seat {seat}")

        action.player_name = player.name
        player.add_action(action)
        self.action_history.append(action)

        # Update pot with the wagered amount.
        if action.action_type in (
            ActionType.BET,
            ActionType.RAISE,
            ActionType.CALL,
            ActionType.ALL_IN,
            ActionType.POST_BLIND,
        ):
            self.pot_size += action.amount

    def reset_for_new_hand(self) -> None:
        """Reset mutable state in preparation for a new hand.

        Preserves table info and player names/seats/stacks but clears all
        hand-specific state (cards, pot, actions, street).
        """
        self.hand_id = str(uuid.uuid4())
        self.hand_number += 1
        self.current_street = Street.PREFLOP
        self.community_cards = []
        self.pot_size = 0.0
        self.side_pots = []
        self.action_history = []
        self.timestamp = time.time()

        for player in self.players:
            player.hole_cards = None
            player.is_active = True
            player.is_all_in = False
            player.current_bet = 0.0
            player.actions = []

    def __str__(self) -> str:
        board = " ".join(str(c) for c in self.community_cards) or "---"
        return (
            f"Hand #{self.hand_number} | {self.current_street.name} | "
            f"Board: {board} | Pot: ${self.pot_size:.2f} | "
            f"Players: {self.get_num_players()}"
        )
