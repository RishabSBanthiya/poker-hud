"""Integration tests for data persistence and retrieval.

Tests that hand data survives database round-trips, stats
are computed correctly from stored data, and the persistence
layer works correctly across simulated app restarts.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    ActionType,
    HandState,
    PlayerAction,
    PlayerState,
    Street,
)
from src.engine.hand_history import HandHistoryParser
from src.stats.connection_manager import ConnectionManager
from src.stats.hand_repository import HandRepository
from src.stats.player_stats_repository import (
    PlayerStats,
    PlayerStatsRepository,
)
from src.stats.stats_aggregator import StatsAggregator

from tests.conftest import make_preflop_only_hand, make_sample_hand_record

# ---------------------------------------------------------------------------
# Hand history parser -> database storage -> stats calculation
# ---------------------------------------------------------------------------


class TestHandHistoryPersistence:
    """Test hand history round-trips through the parser and database."""

    def test_hand_state_to_record_and_back(self) -> None:
        """A HandState round-trips through serialization without data loss."""
        parser = HandHistoryParser()

        hand = HandState(
            hand_id="round-trip-001",
            street=Street.RIVER,
            community_cards=[
                Card(Rank.ACE, Suit.HEARTS),
                Card(Rank.KING, Suit.DIAMONDS),
                Card(Rank.QUEEN, Suit.SPADES),
                Card(Rank.JACK, Suit.CLUBS),
                Card(Rank.TEN, Suit.HEARTS),
            ],
            players=[
                PlayerState(name="Alice", seat_index=0, is_hero=True),
                PlayerState(name="Bob", seat_index=1),
            ],
            actions=[
                PlayerAction("Alice", ActionType.RAISE, 3.0, Street.PREFLOP),
                PlayerAction("Bob", ActionType.CALL, 3.0, Street.PREFLOP),
                PlayerAction("Alice", ActionType.BET, 5.0, Street.FLOP),
                PlayerAction("Bob", ActionType.CALL, 5.0, Street.FLOP),
            ],
            pot=16.0,
            big_blind=1.0,
            is_complete=True,
            winner_name="Alice",
        )

        record = parser.hand_state_to_record(hand, timestamp=1234567890.0)
        assert record.hand_id == "round-trip-001"
        assert record.pot == 16.0
        assert record.winner_name == "Alice"
        assert "Alice" in record.players
        assert "Bob" in record.players

        # Round-trip back to HandState
        restored = parser.record_to_hand_state(record)
        assert restored.hand_id == "round-trip-001"
        assert restored.pot == 16.0
        assert restored.winner_name == "Alice"
        assert restored.is_complete is True
        assert len(restored.actions) == 4
        assert len(restored.community_cards) == 5
        assert len(restored.players) == 2

        # Verify community cards preserved
        ranks = [c.rank for c in restored.community_cards]
        assert Rank.ACE in ranks
        assert Rank.KING in ranks

    def test_hand_record_database_round_trip(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """A HandRecord survives database storage and retrieval."""
        repo = HandRepository(in_memory_db)

        record = make_sample_hand_record(hand_id="db-trip-001", pot=42.5)
        repo.save(record)

        loaded = repo.get_by_id("db-trip-001")
        assert loaded is not None
        assert loaded.hand_id == "db-trip-001"
        assert loaded.pot == 42.5
        assert loaded.winner_name == "Alice"
        assert loaded.big_blind == 1.0
        assert "Alice" in loaded.players
        assert "Bob" in loaded.players
        assert "Charlie" in loaded.players

        # Verify actions JSON is intact
        actions = json.loads(loaded.actions_json)
        assert len(actions) == 7
        raise_actions = [a for a in actions if a["action"] == "raise"]
        assert len(raise_actions) == 1
        assert raise_actions[0]["player"] == "Bob"

    def test_hand_record_persists_community_cards(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Community cards are stored and retrieved correctly."""
        repo = HandRepository(in_memory_db)
        record = make_sample_hand_record()
        repo.save(record)

        loaded = repo.get_by_id("test-001")
        assert loaded is not None
        assert loaded.community_cards_str == "Ah,Kd,Qs"

        # Parse community cards back
        parser = HandHistoryParser()
        hand = parser.record_to_hand_state(loaded)
        assert len(hand.community_cards) == 3
        card_tuples = [(c.rank, c.suit) for c in hand.community_cards]
        assert (Rank.ACE, Suit.HEARTS) in card_tuples
        assert (Rank.KING, Suit.DIAMONDS) in card_tuples
        assert (Rank.QUEEN, Suit.SPADES) in card_tuples

    def test_stats_calculated_from_stored_hands(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Stats are correctly computed after hands are stored in DB."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        # Store 5 hands
        for i in range(5):
            record = make_sample_hand_record(hand_id=f"stats-{i}")
            aggregator.process_hand(record)

        assert hand_repo.count() == 5

        # Verify aggregated stats
        bob = stats_repo.get("Bob")
        assert bob is not None
        assert bob.total_hands == 5
        assert bob.vpip_hands == 5  # Bob raised every hand
        assert bob.pfr_hands == 5
        assert bob.vpip == 100.0
        assert bob.pfr == 100.0

        charlie = stats_repo.get("Charlie")
        assert charlie is not None
        assert charlie.total_hands == 5
        assert charlie.vpip_hands == 0  # Charlie only posted blind + folded
        assert charlie.vpip == 0.0

    def test_mixed_hand_types_stats(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Stats handle a mix of postflop and preflop-only hands."""
        hand_repo = HandRepository(in_memory_db)
        stats_repo = PlayerStatsRepository(in_memory_db)
        aggregator = StatsAggregator(hand_repo, stats_repo)

        # Mix of hand types
        aggregator.process_hand(make_sample_hand_record(hand_id="m1"))
        aggregator.process_hand(make_preflop_only_hand(hand_id="m2"))
        aggregator.process_hand(make_sample_hand_record(hand_id="m3"))

        assert hand_repo.count() == 3

        bob = stats_repo.get("Bob")
        assert bob is not None
        assert bob.total_hands == 3
        assert bob.pfr_hands == 3  # Bob always raises

        alice = stats_repo.get("Alice")
        assert alice is not None
        assert alice.total_hands == 3
        # Alice folds in preflop-only hand, calls in sample hands
        assert alice.vpip_hands == 2


# ---------------------------------------------------------------------------
# Stats survive app restart (load from DB)
# ---------------------------------------------------------------------------


class TestStatsSurviveRestart:
    """Test that statistics persist across simulated app restarts."""

    def test_stats_persist_across_connection_cycles(self) -> None:
        """Stats stored in one session are available in a new session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")

            # Session 1: Create and populate
            cm1 = ConnectionManager(db_path=db_path)
            cm1.initialize()
            hand_repo1 = HandRepository(cm1)
            stats_repo1 = PlayerStatsRepository(cm1)
            aggregator1 = StatsAggregator(hand_repo1, stats_repo1)

            for i in range(3):
                aggregator1.process_hand(
                    make_sample_hand_record(hand_id=f"persist-{i}")
                )

            bob_stats_1 = stats_repo1.get("Bob")
            assert bob_stats_1 is not None
            assert bob_stats_1.total_hands == 3

            cm1.close()

            # Session 2: Reconnect and verify data survived
            cm2 = ConnectionManager(db_path=db_path)
            cm2.initialize()
            hand_repo2 = HandRepository(cm2)
            stats_repo2 = PlayerStatsRepository(cm2)

            assert hand_repo2.count() == 3

            bob_stats_2 = stats_repo2.get("Bob")
            assert bob_stats_2 is not None
            assert bob_stats_2.total_hands == 3
            assert bob_stats_2.pfr_hands == 3
            assert bob_stats_2.vpip_hands == 3

            alice_stats_2 = stats_repo2.get("Alice")
            assert alice_stats_2 is not None
            assert alice_stats_2.total_hands == 3

            cm2.close()

    def test_new_hands_accumulate_with_persisted_stats(self) -> None:
        """New hands in a new session add to previously stored stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "accumulate.db")

            # Session 1
            cm1 = ConnectionManager(db_path=db_path)
            cm1.initialize()
            agg1 = StatsAggregator(
                HandRepository(cm1), PlayerStatsRepository(cm1)
            )
            agg1.process_hand(make_sample_hand_record(hand_id="s1-h1"))
            agg1.process_hand(make_sample_hand_record(hand_id="s1-h2"))
            cm1.close()

            # Session 2
            cm2 = ConnectionManager(db_path=db_path)
            cm2.initialize()
            stats_repo2 = PlayerStatsRepository(cm2)
            agg2 = StatsAggregator(HandRepository(cm2), stats_repo2)

            # Process more hands
            agg2.process_hand(make_sample_hand_record(hand_id="s2-h1"))

            bob = stats_repo2.get("Bob")
            assert bob is not None
            assert bob.total_hands == 3  # 2 from session 1 + 1 from session 2

            cm2.close()

    def test_hand_records_retrievable_after_restart(self) -> None:
        """Individual hand records can be retrieved after a restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "hands.db")

            # Session 1: store hands
            cm1 = ConnectionManager(db_path=db_path)
            cm1.initialize()
            repo1 = HandRepository(cm1)
            repo1.save(make_sample_hand_record(hand_id="lookup-1", pot=25.0))
            repo1.save(make_preflop_only_hand(hand_id="lookup-2"))
            cm1.close()

            # Session 2: retrieve
            cm2 = ConnectionManager(db_path=db_path)
            cm2.initialize()
            repo2 = HandRepository(cm2)

            h1 = repo2.get_by_id("lookup-1")
            assert h1 is not None
            assert h1.pot == 25.0
            assert h1.community_cards_str == "Ah,Kd,Qs"

            h2 = repo2.get_by_id("lookup-2")
            assert h2 is not None
            assert h2.community_cards_str == ""

            all_hands = repo2.get_all()
            assert len(all_hands) == 2

            cm2.close()


# ---------------------------------------------------------------------------
# Repository operations
# ---------------------------------------------------------------------------


class TestRepositoryOperations:
    """Test repository CRUD operations."""

    def test_hand_repo_get_by_player(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Hands can be filtered by player name."""
        repo = HandRepository(in_memory_db)

        repo.save(
            make_sample_hand_record(
                hand_id="p1", players=["Alice", "Bob"]
            )
        )
        repo.save(
            make_sample_hand_record(
                hand_id="p2", players=["Bob", "Charlie"]
            )
        )
        repo.save(
            make_sample_hand_record(
                hand_id="p3", players=["Alice", "Dave"]
            )
        )

        alice_hands = repo.get_by_player("Alice")
        assert len(alice_hands) == 2

        bob_hands = repo.get_by_player("Bob")
        assert len(bob_hands) == 2

        dave_hands = repo.get_by_player("Dave")
        assert len(dave_hands) == 1

    def test_hand_repo_delete(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """Hands can be deleted by ID."""
        repo = HandRepository(in_memory_db)
        repo.save(make_sample_hand_record(hand_id="del-1"))
        repo.save(make_sample_hand_record(hand_id="del-2"))

        assert repo.count() == 2
        assert repo.delete("del-1") is True
        assert repo.count() == 1
        assert repo.get_by_id("del-1") is None
        assert repo.delete("nonexistent") is False

    def test_player_stats_repo_get_all(
        self, in_memory_db: ConnectionManager
    ) -> None:
        """All player stats can be retrieved."""
        repo = PlayerStatsRepository(in_memory_db)

        repo.save(PlayerStats(player_name="Alice", total_hands=10, vpip_hands=5))
        repo.save(PlayerStats(player_name="Bob", total_hands=20, vpip_hands=15))

        all_stats = repo.get_all()
        assert len(all_stats) == 2
        names = {s.player_name for s in all_stats}
        assert names == {"Alice", "Bob"}

    def test_player_stats_computed_properties(self) -> None:
        """PlayerStats computed properties return correct percentages."""
        stats = PlayerStats(
            player_name="Test",
            total_hands=100,
            vpip_hands=25,
            pfr_hands=18,
            three_bet_opportunities=50,
            three_bet_hands=4,
            cbet_opportunities=30,
            cbet_hands=21,
            total_bets=40,
            total_raises=20,
            total_calls=30,
            total_folds=50,
            went_to_showdown=15,
            showdown_opportunities=25,
        )

        assert stats.vpip == 25.0
        assert stats.pfr == 18.0
        assert stats.three_bet_pct == 8.0
        assert stats.cbet_pct == 70.0
        assert stats.aggression_factor == 2.0  # (40+20)/30
        assert stats.wtsd == 60.0  # 15/25*100

    def test_player_stats_zero_division_safe(self) -> None:
        """PlayerStats handles zero-division cases gracefully."""
        stats = PlayerStats(player_name="New")

        assert stats.vpip == 0.0
        assert stats.pfr == 0.0
        assert stats.three_bet_pct == 0.0
        assert stats.cbet_pct == 0.0
        assert stats.aggression_factor == 0.0  # no bets, no calls
        assert stats.wtsd == 0.0


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class TestConnectionManager:
    """Test database connection management."""

    def test_in_memory_database(self) -> None:
        """In-memory database initializes correctly."""
        cm = ConnectionManager(db_path=":memory:")
        cm.initialize()
        assert cm.is_initialized
        conn = cm.get_connection()
        assert conn is not None
        cm.close()

    def test_file_database_creation(self) -> None:
        """File-based database is created with parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "subdir" / "test.db")
            cm = ConnectionManager(db_path=db_path)
            cm.initialize()
            assert cm.is_initialized
            assert Path(db_path).exists()
            cm.close()

    def test_schema_creates_tables(self) -> None:
        """Schema initialization creates expected tables."""
        cm = ConnectionManager(db_path=":memory:")
        cm.initialize()
        conn = cm.get_connection()

        # Check hands table exists
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hands'"
        ).fetchone()
        assert result is not None

        # Check player_stats table exists
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='player_stats'"
        ).fetchone()
        assert result is not None

        cm.close()
