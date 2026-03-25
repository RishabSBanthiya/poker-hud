"""Shared test fixtures for the Poker HUD test suite."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from src.detection.card import Card, DetectedCard, Rank, Suit
from src.detection.card_recognition import CardRecognitionResult
from src.detection.detection_pipeline import DetectionResult, PlayerInfo
from src.engine.hand_history import HandRecord
from src.stats.connection_manager import ConnectionManager
from src.stats.hand_repository import HandRepository
from src.stats.player_stats_repository import PlayerStatsRepository

# ---------------------------------------------------------------------------
# Card fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ace_spades() -> Card:
    """Ace of spades."""
    return Card(rank=Rank.ACE, suit=Suit.SPADES)


@pytest.fixture()
def king_hearts() -> Card:
    """King of hearts."""
    return Card(rank=Rank.KING, suit=Suit.HEARTS)


@pytest.fixture()
def queen_diamonds() -> Card:
    """Queen of diamonds."""
    return Card(rank=Rank.QUEEN, suit=Suit.DIAMONDS)


# ---------------------------------------------------------------------------
# Frame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def green_felt_frame() -> np.ndarray:
    """A 600x800 green felt poker table frame (BGR)."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:] = (34, 120, 50)  # Green felt in BGR
    return frame


@pytest.fixture()
def blank_frame() -> np.ndarray:
    """A 480x640 black frame (BGR)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Detection fixtures
# ---------------------------------------------------------------------------


def make_detected_card(
    rank: Rank,
    suit: Suit,
    x: int = 0,
    y: int = 0,
    confidence: float = 0.95,
) -> DetectedCard:
    """Create a DetectedCard with default bounding box."""
    return DetectedCard(
        rank=rank,
        suit=suit,
        confidence=confidence,
        bounding_box=(x, y, 60, 80),
    )


def make_detection_result(
    community_cards: list[DetectedCard] | None = None,
    hole_cards: list[DetectedCard] | None = None,
    players: list[PlayerInfo] | None = None,
) -> DetectionResult:
    """Create a DetectionResult with given cards and players."""
    card_result = CardRecognitionResult(
        community_cards=community_cards or [],
        hole_cards=hole_cards or [],
        all_detections=(community_cards or []) + (hole_cards or []),
        frame_timestamp=time.time(),
    )
    return DetectionResult(
        card_result=card_result,
        players=players or [],
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Hand history fixtures
# ---------------------------------------------------------------------------


def make_sample_hand_record(
    hand_id: str = "test-001",
    players: list[str] | None = None,
    winner: str = "Alice",
    pot: float = 10.0,
    big_blind: float = 1.0,
) -> HandRecord:
    """Create a sample hand record with a realistic action sequence.

    A preflop hand where Alice raises, Bob calls, Charlie folds,
    then flop action: Alice bets, Bob calls. Alice wins.
    """
    players = players or ["Alice", "Bob", "Charlie"]

    actions = [
        {
            "player": "Charlie",
            "action": "post_blind",
            "amount": 0.5,
            "street": "preflop",
        },
        {"player": "Alice", "action": "post_blind", "amount": 1.0, "street": "preflop"},
        {"player": "Bob", "action": "raise", "amount": 3.0, "street": "preflop"},
        {"player": "Charlie", "action": "fold", "amount": 0.0, "street": "preflop"},
        {"player": "Alice", "action": "call", "amount": 2.0, "street": "preflop"},
        {"player": "Alice", "action": "bet", "amount": 4.0, "street": "flop"},
        {"player": "Bob", "action": "call", "amount": 4.0, "street": "flop"},
    ]

    return HandRecord(
        hand_id=hand_id,
        players=players,
        actions_json=json.dumps(actions),
        community_cards_str="Ah,Kd,Qs",
        pot=pot,
        winner_name=winner,
        big_blind=big_blind,
        timestamp=time.time(),
    )


def make_preflop_only_hand(
    hand_id: str = "pf-001",
    players: list[str] | None = None,
) -> HandRecord:
    """Create a hand that ends preflop (everyone folds to a raise)."""
    players = players or ["Alice", "Bob", "Charlie"]

    actions = [
        {
            "player": "Charlie",
            "action": "post_blind",
            "amount": 0.5,
            "street": "preflop",
        },
        {"player": "Alice", "action": "post_blind", "amount": 1.0, "street": "preflop"},
        {"player": "Bob", "action": "raise", "amount": 3.0, "street": "preflop"},
        {"player": "Charlie", "action": "fold", "amount": 0.0, "street": "preflop"},
        {"player": "Alice", "action": "fold", "amount": 0.0, "street": "preflop"},
    ]

    return HandRecord(
        hand_id=hand_id,
        players=players,
        actions_json=json.dumps(actions),
        community_cards_str="",
        pot=4.5,
        winner_name="Bob",
        big_blind=1.0,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def in_memory_db() -> ConnectionManager:
    """Create an in-memory SQLite database with schema initialized."""
    cm = ConnectionManager(db_path=":memory:")
    cm.initialize()
    return cm


@pytest.fixture()
def hand_repo(in_memory_db: ConnectionManager) -> HandRepository:
    """Hand repository backed by in-memory database."""
    return HandRepository(in_memory_db)


@pytest.fixture()
def stats_repo(in_memory_db: ConnectionManager) -> PlayerStatsRepository:
    """Player stats repository backed by in-memory database."""
    return PlayerStatsRepository(in_memory_db)
