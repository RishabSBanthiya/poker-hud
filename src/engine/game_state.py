"""Game state data models for poker hand tracking.

Defines the core data structures representing the state of a poker
hand: players, actions, streets, and the overall game state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.detection.card import Card


class Street(Enum):
    """Poker hand streets (phases)."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(Enum):
    """Types of player actions."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_BLIND = "post_blind"


@dataclass
class PlayerAction:
    """A single player action within a hand.

    Attributes:
        player_name: Name of the acting player.
        action_type: Type of action taken.
        amount: Bet/raise amount (0 for fold/check).
        street: Which street the action occurred on.
    """

    player_name: str
    action_type: ActionType
    amount: float = 0.0
    street: Street = Street.PREFLOP


@dataclass
class PlayerState:
    """State of a single player in the current hand.

    Attributes:
        name: Player name.
        seat_index: Seat position at the table.
        stack: Current chip stack.
        hole_cards: Player's hole cards (if visible).
        is_active: Whether the player is still in the hand.
        is_hero: Whether this is the local player.
        total_bet: Total amount bet this hand.
    """

    name: str
    seat_index: int
    stack: float = 0.0
    hole_cards: list[Card] = field(default_factory=list)
    is_active: bool = True
    is_hero: bool = False
    total_bet: float = 0.0


@dataclass
class HandState:
    """Complete state of a single poker hand.

    Attributes:
        hand_id: Unique identifier for this hand.
        street: Current street.
        community_cards: Cards on the board.
        players: List of player states.
        actions: Chronological list of all actions.
        pot: Current pot size.
        dealer_seat: Seat index of the dealer button.
        small_blind: Small blind amount.
        big_blind: Big blind amount.
        is_complete: Whether the hand has finished.
        winner_name: Name of the hand winner (if complete).
    """

    hand_id: str = ""
    street: Street = Street.PREFLOP
    community_cards: list[Card] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=list)
    actions: list[PlayerAction] = field(default_factory=list)
    pot: float = 0.0
    dealer_seat: int = 0
    small_blind: float = 0.5
    big_blind: float = 1.0
    is_complete: bool = False
    winner_name: Optional[str] = None

    def get_player(self, name: str) -> Optional[PlayerState]:
        """Find a player by name.

        Args:
            name: Player name to look up.

        Returns:
            PlayerState if found, None otherwise.
        """
        for p in self.players:
            if p.name == name:
                return p
        return None

    def get_active_players(self) -> list[PlayerState]:
        """Return all players still active in the hand."""
        return [p for p in self.players if p.is_active]

    def get_actions_for_street(self, street: Street) -> list[PlayerAction]:
        """Return all actions on a specific street.

        Args:
            street: The street to filter by.

        Returns:
            List of actions on that street.
        """
        return [a for a in self.actions if a.street == street]
