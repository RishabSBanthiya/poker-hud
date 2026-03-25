"""Tests for stats.repository -- Player, Hand, and Action repositories."""

from __future__ import annotations

import sqlite3

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
from src.stats.database import DatabaseManager
from src.stats.repository import ActionRepository, HandRepository, PlayerRepository


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory database, fully initialized."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON;")
    c.row_factory = sqlite3.Row
    DatabaseManager(c).initialize()
    return c


@pytest.fixture()
def player_repo(conn: sqlite3.Connection) -> PlayerRepository:
    return PlayerRepository(conn)


@pytest.fixture()
def hand_repo(conn: sqlite3.Connection) -> HandRepository:
    return HandRepository(conn)


@pytest.fixture()
def action_repo(conn: sqlite3.Connection) -> ActionRepository:
    return ActionRepository(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game_state() -> GameState:
    """Build a realistic GameState for testing hand persistence."""
    gs = GameState(
        table_name="Test Table",
        small_blind=1.0,
        big_blind=2.0,
        ante=0.0,
        hand_number=42,
        hero_seat=0,
        current_street=Street.RIVER,
        community_cards=[
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
            Card(Rank.TEN, Suit.CLUBS),
            Card(Rank.JACK, Suit.SPADES),
        ],
    )

    alice = Player(
        name="Alice",
        seat_number=0,
        position=Position.SB,
        stack_size=98.0,
        hole_cards=[Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)],
    )
    bob = Player(
        name="Bob",
        seat_number=1,
        position=Position.BB,
        stack_size=96.0,
    )
    gs.players = [alice, bob]

    # Record some actions through the GameState API.
    gs.record_action(
        0,
        PlayerAction(
            action_type=ActionType.POST_BLIND,
            amount=1.0,
            street=Street.PREFLOP,
        ),
    )
    gs.record_action(
        1,
        PlayerAction(
            action_type=ActionType.POST_BLIND,
            amount=2.0,
            street=Street.PREFLOP,
        ),
    )
    gs.record_action(
        0,
        PlayerAction(
            action_type=ActionType.RAISE,
            amount=6.0,
            street=Street.PREFLOP,
        ),
    )
    gs.record_action(
        1,
        PlayerAction(
            action_type=ActionType.CALL,
            amount=4.0,
            street=Street.PREFLOP,
        ),
    )
    gs.record_action(
        0,
        PlayerAction(
            action_type=ActionType.BET,
            amount=8.0,
            street=Street.FLOP,
        ),
    )
    gs.record_action(
        1,
        PlayerAction(
            action_type=ActionType.FOLD,
            amount=0.0,
            street=Street.FLOP,
        ),
    )

    return gs


# ===========================================================================
# PlayerRepository
# ===========================================================================


class TestPlayerRepository:
    """Tests for PlayerRepository CRUD operations."""

    def test_get_or_create_new(self, player_repo: PlayerRepository) -> None:
        p = player_repo.get_or_create("Alice")
        assert p["name"] == "Alice"
        assert p["total_hands"] == 0

    def test_get_or_create_existing(
        self, player_repo: PlayerRepository
    ) -> None:
        p1 = player_repo.get_or_create("Alice")
        p2 = player_repo.get_or_create("Alice")
        assert p1["id"] == p2["id"]

    def test_get_by_name_missing(
        self, player_repo: PlayerRepository
    ) -> None:
        assert player_repo.get_by_name("Nobody") is None

    def test_get_by_name_found(
        self, player_repo: PlayerRepository
    ) -> None:
        player_repo.get_or_create("Bob")
        result = player_repo.get_by_name("Bob")
        assert result is not None
        assert result["name"] == "Bob"

    def test_update(self, player_repo: PlayerRepository) -> None:
        p = player_repo.get_or_create("Carol")
        p["total_hands"] = 99
        p["notes"] = "tight player"
        player_repo.update(p)

        refreshed = player_repo.get_by_name("Carol")
        assert refreshed is not None
        assert refreshed["total_hands"] == 99
        assert refreshed["notes"] == "tight player"

    def test_get_all(self, player_repo: PlayerRepository) -> None:
        player_repo.get_or_create("Zara")
        player_repo.get_or_create("Alice")
        all_players = player_repo.get_all()
        names = [p["name"] for p in all_players]
        assert names == ["Alice", "Zara"]  # alphabetical

    def test_unique_name_constraint(
        self, player_repo: PlayerRepository, conn: sqlite3.Connection
    ) -> None:
        player_repo.get_or_create("Alice")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO players (name) VALUES ('Alice')")


# ===========================================================================
# HandRepository
# ===========================================================================


class TestHandRepository:
    """Tests for HandRepository persistence and queries."""

    def test_save_hand_returns_id(
        self, hand_repo: HandRepository
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)
        assert isinstance(hand_id, int)
        assert hand_id >= 1

    def test_get_hand(self, hand_repo: HandRepository) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        hand = hand_repo.get_hand(hand_id)
        assert hand is not None
        assert hand["hand_number"] == 42
        assert hand["table_name"] == "Test Table"
        assert hand["small_blind"] == 1.0
        assert hand["big_blind"] == 2.0

    def test_get_hand_not_found(self, hand_repo: HandRepository) -> None:
        assert hand_repo.get_hand(9999) is None

    def test_get_hands_for_player(
        self, hand_repo: HandRepository
    ) -> None:
        gs = _make_game_state()
        hand_repo.save_hand(gs)

        hands = hand_repo.get_hands_for_player("Alice")
        assert len(hands) == 1
        assert hands[0]["hand_number"] == 42

    def test_get_hands_for_player_missing(
        self, hand_repo: HandRepository
    ) -> None:
        assert hand_repo.get_hands_for_player("Nobody") == []

    def test_get_recent_hands(self, hand_repo: HandRepository) -> None:
        gs = _make_game_state()
        hand_repo.save_hand(gs)

        recent = hand_repo.get_recent_hands(limit=10)
        assert len(recent) == 1

    def test_save_hand_creates_players(
        self,
        hand_repo: HandRepository,
        player_repo: PlayerRepository,
    ) -> None:
        gs = _make_game_state()
        hand_repo.save_hand(gs)

        alice = player_repo.get_by_name("Alice")
        bob = player_repo.get_by_name("Bob")
        assert alice is not None
        assert bob is not None
        assert alice["total_hands"] == 1
        assert bob["total_hands"] == 1

    def test_save_hand_persists_actions(
        self,
        hand_repo: HandRepository,
        action_repo: ActionRepository,
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        actions = action_repo.get_actions_for_hand(hand_id)
        assert len(actions) == 6
        assert actions[0]["action_type"] == "post_blind"
        assert actions[0]["player_name"] == "Alice"

    def test_save_hand_persists_community_cards(
        self,
        hand_repo: HandRepository,
        conn: sqlite3.Connection,
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        cur = conn.execute(
            "SELECT * FROM community_cards WHERE hand_id = ? ORDER BY street",
            (hand_id,),
        )
        rows = cur.fetchall()
        streets = [dict(r)["street"] for r in rows]
        assert "FLOP" in streets
        assert "TURN" in streets
        assert "RIVER" in streets

    def test_save_hand_limit_param(
        self, hand_repo: HandRepository
    ) -> None:
        gs = _make_game_state()
        hand_repo.save_hand(gs)
        gs.hand_number = 43
        gs.hand_id = "second-hand"
        hand_repo.save_hand(gs)

        one = hand_repo.get_hands_for_player("Alice", limit=1)
        assert len(one) == 1

    def test_save_hand_persists_hand_players(
        self,
        hand_repo: HandRepository,
        conn: sqlite3.Connection,
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        cur = conn.execute(
            "SELECT * FROM hand_players WHERE hand_id = ? ORDER BY seat",
            (hand_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        assert len(rows) == 2
        assert rows[0]["seat"] == 0
        assert rows[0]["position"] == "SB"
        assert rows[1]["seat"] == 1
        assert rows[1]["position"] == "BB"

    def test_save_hand_records_hole_cards(
        self,
        hand_repo: HandRepository,
        conn: sqlite3.Connection,
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        cur = conn.execute(
            "SELECT hole_cards FROM hand_players WHERE hand_id = ? ORDER BY seat",
            (hand_id,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        # Alice has hole cards, Bob does not
        assert rows[0]["hole_cards"] != ""
        assert rows[1]["hole_cards"] == ""


# ===========================================================================
# ActionRepository
# ===========================================================================


class TestActionRepository:
    """Tests for ActionRepository query operations."""

    def test_get_actions_for_hand(
        self,
        hand_repo: HandRepository,
        action_repo: ActionRepository,
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        actions = action_repo.get_actions_for_hand(hand_id)
        assert len(actions) == 6
        # Check ordering by sequence number.
        seqs = [a["sequence_number"] for a in actions]
        assert seqs == sorted(seqs)

    def test_get_actions_for_hand_empty(
        self, action_repo: ActionRepository
    ) -> None:
        assert action_repo.get_actions_for_hand(9999) == []

    def test_get_actions_for_player(
        self,
        hand_repo: HandRepository,
        action_repo: ActionRepository,
    ) -> None:
        gs = _make_game_state()
        hand_repo.save_hand(gs)

        alice_actions = action_repo.get_actions_for_player("Alice")
        assert len(alice_actions) == 3  # post_blind, raise, bet

        bob_actions = action_repo.get_actions_for_player("Bob")
        assert len(bob_actions) == 3  # post_blind, call, fold

    def test_get_actions_for_player_by_street(
        self,
        hand_repo: HandRepository,
        action_repo: ActionRepository,
    ) -> None:
        gs = _make_game_state()
        hand_repo.save_hand(gs)

        preflop = action_repo.get_actions_for_player("Alice", street="PREFLOP")
        assert len(preflop) == 2  # post_blind, raise

        flop = action_repo.get_actions_for_player("Alice", street="FLOP")
        assert len(flop) == 1  # bet

    def test_get_actions_for_missing_player(
        self, action_repo: ActionRepository
    ) -> None:
        assert action_repo.get_actions_for_player("Nobody") == []

    def test_actions_include_player_name(
        self,
        hand_repo: HandRepository,
        action_repo: ActionRepository,
    ) -> None:
        gs = _make_game_state()
        hand_id = hand_repo.save_hand(gs)

        actions = action_repo.get_actions_for_hand(hand_id)
        for action in actions:
            assert "player_name" in action
            assert action["player_name"] in ("Alice", "Bob")
