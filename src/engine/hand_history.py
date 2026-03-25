"""Hand history parsing and serialization.

Converts completed hand state objects to/from a storable format
for database persistence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    ActionType,
    GameState,
    Player,
    PlayerAction,
    Street,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight hand state types used by the integration wrappers
# ---------------------------------------------------------------------------


@dataclass
class PlayerState:
    """Simplified player state for hand history tracking.

    Attributes:
        name: Player screen name.
        seat_index: Seat position (0-based).
        is_hero: Whether this player is the user.
        is_active: Whether the player is still in the hand.
        hole_cards: Player's private cards, if known.
        stack: Current stack size.
    """

    name: str
    seat_index: int = 0
    is_hero: bool = False
    is_active: bool = True
    hole_cards: list[Card] = field(default_factory=list)
    stack: float = 0.0


@dataclass
class HandState:
    """Simplified hand state for tracking a hand in progress.

    Attributes:
        hand_id: Unique identifier for this hand.
        street: Current betting street.
        community_cards: Board cards.
        players: Players at the table.
        actions: Chronological list of actions.
        pot: Current pot size.
        big_blind: Big blind amount.
        is_complete: Whether the hand has finished.
        winner_name: Name of the winner (if known).
    """

    hand_id: str = ""
    street: Street = Street.PREFLOP
    community_cards: list[Card] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=list)
    actions: list[PlayerAction] = field(default_factory=list)
    pot: float = 0.0
    big_blind: float = 1.0
    is_complete: bool = False
    winner_name: Optional[str] = None

    def get_player(self, name: str) -> Optional[PlayerState]:
        """Look up a player by name.

        Args:
            name: Player name to search for.

        Returns:
            The matching PlayerState, or None.
        """
        for p in self.players:
            if p.name == name:
                return p
        return None

    def get_active_players(self) -> list[PlayerState]:
        """Return all players still in the hand.

        Returns:
            List of active PlayerState objects.
        """
        return [p for p in self.players if p.is_active]


@dataclass
class HandRecord:
    """A serializable record of a completed hand.

    Attributes:
        hand_id: Unique hand identifier.
        players: List of player name strings.
        actions_json: JSON-serialized list of actions.
        community_cards_str: Comma-separated card strings (e.g., "Ah,Kd,Qs").
        pot: Final pot size.
        winner_name: Name of the winner.
        big_blind: Big blind amount.
        timestamp: Unix timestamp of hand completion.
    """

    hand_id: str
    players: list[str]
    actions_json: str
    community_cards_str: str
    pot: float
    winner_name: str
    big_blind: float
    timestamp: float = 0.0


class HandHistoryParser:
    """Converts between HandState and HandRecord formats."""

    @staticmethod
    def hand_state_to_record(
        hand: HandState, timestamp: float = 0.0
    ) -> HandRecord:
        """Convert a completed HandState to a storable HandRecord.

        Args:
            hand: The completed hand state.
            timestamp: When the hand was completed.

        Returns:
            A HandRecord suitable for database storage.
        """
        actions_data = [
            {
                "player": a.player_name,
                "action": a.action_type.value,
                "amount": a.amount,
                "street": a.street.value if hasattr(a.street, 'value') else str(a.street),
            }
            for a in hand.actions
        ]

        community_str = ",".join(
            f"{c.rank.value}{c.suit.value[0]}" for c in hand.community_cards
        )

        return HandRecord(
            hand_id=hand.hand_id,
            players=[p.name for p in hand.players],
            actions_json=json.dumps(actions_data),
            community_cards_str=community_str,
            pot=hand.pot,
            winner_name=hand.winner_name or "",
            big_blind=hand.big_blind,
            timestamp=timestamp,
        )

    @staticmethod
    def record_to_hand_state(record: HandRecord) -> HandState:
        """Convert a HandRecord back to a HandState.

        Args:
            record: The stored hand record.

        Returns:
            A reconstructed HandState.
        """
        actions_data = json.loads(record.actions_json)
        actions = [
            PlayerAction(
                player_name=a["player"],
                action_type=ActionType(a["action"]),
                amount=a["amount"],
                street=Street[a["street"].upper()] if isinstance(a["street"], str) else a["street"],
            )
            for a in actions_data
        ]

        players = [
            PlayerState(name=name, seat_index=i)
            for i, name in enumerate(record.players)
        ]

        # Parse community cards
        community_cards: list[Card] = []
        if record.community_cards_str:
            for card_str in record.community_cards_str.split(","):
                card_str = card_str.strip()
                if not card_str:
                    continue
                card = HandHistoryParser._parse_card_str(card_str)
                if card is not None:
                    community_cards.append(card)

        return HandState(
            hand_id=record.hand_id,
            street=Street.SHOWDOWN,
            community_cards=community_cards,
            players=players,
            actions=actions,
            pot=record.pot,
            big_blind=record.big_blind,
            is_complete=True,
            winner_name=record.winner_name or None,
        )

    @staticmethod
    def _parse_card_str(card_str: str) -> Card | None:
        """Parse a short card string like 'Ah' or '10d'.

        Args:
            card_str: Short card notation.

        Returns:
            A Card, or None if parsing fails.
        """
        suit_char_map = {
            "h": Suit.HEARTS,
            "d": Suit.DIAMONDS,
            "c": Suit.CLUBS,
            "s": Suit.SPADES,
        }
        rank_map = {r.value: r for r in Rank}

        if len(card_str) < 2:
            return None

        suit_char = card_str[-1].lower()
        rank_str = card_str[:-1]

        suit = suit_char_map.get(suit_char)
        rank = rank_map.get(rank_str)

        if suit is None or rank is None:
            logger.warning("Failed to parse card string: %s", card_str)
            return None

        return Card(rank=rank, suit=suit)
