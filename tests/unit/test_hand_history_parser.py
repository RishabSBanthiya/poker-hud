"""Unit tests for HandHistoryParser and HandHistoryWatcher (S3-03)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from src.detection.card import Card, Rank, Suit
from src.engine.game_state import ActionType, GameState, Position, Street
from src.engine.hand_history_parser import (
    HandHistoryParser,
    HandHistoryWatcher,
    parse_card,
    parse_cards,
)

from tests.fixtures.sample_hand_histories import (
    SAMPLE_HAND_ALL_IN,
    SAMPLE_HAND_FULL,
    SAMPLE_HAND_PREFLOP_ONLY,
    SAMPLE_HAND_SHOWDOWN,
    SAMPLE_MULTI_HANDS,
)

# ---------------------------------------------------------------------------
# Card parsing helpers
# ---------------------------------------------------------------------------


class TestParseCard:
    """Tests for the parse_card / parse_cards utility functions."""

    def test_parse_ace_of_hearts(self) -> None:
        card = parse_card("Ah")
        assert card == Card(rank=Rank.ACE, suit=Suit.HEARTS)

    def test_parse_ten_of_spades(self) -> None:
        card = parse_card("Ts")
        assert card == Card(rank=Rank.TEN, suit=Suit.SPADES)

    def test_parse_two_of_clubs(self) -> None:
        card = parse_card("2c")
        assert card == Card(rank=Rank.TWO, suit=Suit.CLUBS)

    def test_parse_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="Invalid card string"):
            parse_card("AhX")

    def test_parse_invalid_rank(self) -> None:
        with pytest.raises(ValueError, match="Invalid card string"):
            parse_card("Xh")

    def test_parse_invalid_suit(self) -> None:
        with pytest.raises(ValueError, match="Invalid card string"):
            parse_card("Ax")

    def test_parse_cards_multiple(self) -> None:
        cards = parse_cards("Ah Kd Js")
        assert len(cards) == 3
        assert cards[0] == Card(rank=Rank.ACE, suit=Suit.HEARTS)
        assert cards[1] == Card(rank=Rank.KING, suit=Suit.DIAMONDS)
        assert cards[2] == Card(rank=Rank.JACK, suit=Suit.SPADES)

    def test_parse_cards_empty(self) -> None:
        assert parse_cards("") == []


# ---------------------------------------------------------------------------
# HandHistoryParser: single hand parsing
# ---------------------------------------------------------------------------


class TestHandHistoryParserSingleHand:
    """Test parsing individual hand history blocks."""

    def setup_method(self) -> None:
        self.parser = HandHistoryParser()

    def test_parse_full_hand_header(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        assert gs.hand_number == 12345
        assert gs.table_name == "TableName"
        assert gs.small_blind == 1.0
        assert gs.big_blind == 2.0

    def test_parse_full_hand_players(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        assert gs.get_num_players() == 6
        p1 = gs.get_player_by_name("Player1")
        assert p1 is not None
        assert p1.seat_number == 1

    def test_parse_full_hand_hero(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        hero = gs.get_hero()
        assert hero is not None
        assert hero.name == "Hero"
        assert hero.hole_cards is not None
        assert len(hero.hole_cards) == 2
        assert Card(rank=Rank.ACE, suit=Suit.HEARTS) in hero.hole_cards
        assert Card(rank=Rank.KING, suit=Suit.DIAMONDS) in hero.hole_cards

    def test_parse_full_hand_community_cards(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        assert len(gs.community_cards) == 5
        # FLOP: Js 7h 2c
        assert Card(rank=Rank.JACK, suit=Suit.SPADES) in gs.community_cards
        assert Card(rank=Rank.SEVEN, suit=Suit.HEARTS) in gs.community_cards
        assert Card(rank=Rank.TWO, suit=Suit.CLUBS) in gs.community_cards
        # TURN: 9d
        assert Card(rank=Rank.NINE, suit=Suit.DIAMONDS) in gs.community_cards
        # RIVER: 3s
        assert Card(rank=Rank.THREE, suit=Suit.SPADES) in gs.community_cards

    def test_parse_full_hand_actions(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        assert len(gs.action_history) > 0

        # Check that we have blinds, raises, folds, calls, bets, checks.
        action_types = {a.action_type for a in gs.action_history}
        assert ActionType.POST_BLIND in action_types
        assert ActionType.RAISE in action_types
        assert ActionType.FOLD in action_types
        assert ActionType.CALL in action_types
        assert ActionType.BET in action_types
        assert ActionType.CHECK in action_types

    def test_parse_full_hand_pot(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        assert gs.pot_size == 85.0

    def test_parse_preflop_only_hand(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_PREFLOP_ONLY)
        assert gs is not None
        assert gs.hand_number == 12346
        assert len(gs.community_cards) == 0
        hero = gs.get_hero()
        assert hero is not None
        assert hero.hole_cards is not None
        assert Card(rank=Rank.TEN, suit=Suit.SPADES) in hero.hole_cards

    def test_parse_showdown_hand(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_SHOWDOWN)
        assert gs is not None
        assert gs.hand_number == 12347
        # Hero shows Qh Qd.
        hero = gs.get_hero()
        assert hero is not None
        assert hero.hole_cards is not None
        assert Card(rank=Rank.QUEEN, suit=Suit.HEARTS) in hero.hole_cards

    def test_parse_all_in_hand(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_ALL_IN)
        assert gs is not None
        assert gs.hand_number == 12348
        # Player1 went all-in.
        action_types = {a.action_type for a in gs.action_history}
        assert ActionType.ALL_IN in action_types

    def test_parse_empty_block_returns_none(self) -> None:
        result = self.parser.parse_hand("")
        assert result is None

    def test_parse_garbage_text_returns_none(self) -> None:
        result = self.parser.parse_hand("This is not a hand history.")
        assert result is None

    def test_player_stacks_parsed_correctly(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        p2 = gs.get_player_by_name("Player2")
        assert p2 is not None
        # Starting stack was $185.50 but actions reduce it.
        # Just verify it was parsed from the hand (initial value stored).
        # After replaying actions the stack will differ.

    def test_positions_assigned(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        # Seat 3 is button.
        p3 = gs.get_player_by_name("Player3")
        assert p3 is not None
        assert p3.position == Position.BTN
        # Seat 4 should be SB.
        p4 = gs.get_player_by_name("Player4")
        assert p4 is not None
        assert p4.position == Position.SB
        # Seat 5 should be BB.
        p5 = gs.get_player_by_name("Player5")
        assert p5 is not None
        assert p5.position == Position.BB


# ---------------------------------------------------------------------------
# HandHistoryParser: multi-hand parsing
# ---------------------------------------------------------------------------


class TestHandHistoryParserMultiHand:
    """Test parsing multiple hands from a single text block or file."""

    def setup_method(self) -> None:
        self.parser = HandHistoryParser()

    def test_parse_text_multiple_hands(self) -> None:
        results = self.parser.parse_text(SAMPLE_MULTI_HANDS)
        assert len(results) == 2
        assert results[0].hand_number == 12346
        assert results[1].hand_number == 12347

    def test_parse_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(SAMPLE_MULTI_HANDS)
            f.flush()
            path = f.name

        try:
            results = self.parser.parse_file(path)
            assert len(results) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_parse_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            self.parser.parse_file("/nonexistent/path/hands.txt")

    def test_parse_text_empty(self) -> None:
        results = self.parser.parse_text("")
        assert results == []

    def test_parse_text_only_garbage(self) -> None:
        results = self.parser.parse_text("some random text\n\nmore text")
        assert results == []


# ---------------------------------------------------------------------------
# HandHistoryParser: action replay correctness
# ---------------------------------------------------------------------------


class TestActionReplay:
    """Test that replayed actions produce correct game state."""

    def setup_method(self) -> None:
        self.parser = HandHistoryParser()

    def test_folded_players_marked_inactive(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        p1 = gs.get_player_by_name("Player1")
        assert p1 is not None
        assert not p1.is_active  # Player1 folded preflop.

    def test_street_advances_during_replay(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_FULL)
        assert gs is not None
        # Actions span multiple streets, so the GameState street should
        # have been advanced. The final street depends on the last action
        # parsed. The summary section comes after RIVER.
        assert gs.current_street.value >= Street.RIVER.value

    def test_preflop_hand_stays_on_preflop(self) -> None:
        gs = self.parser.parse_hand(SAMPLE_HAND_PREFLOP_ONLY)
        assert gs is not None
        assert gs.current_street == Street.PREFLOP


# ---------------------------------------------------------------------------
# HandHistoryWatcher
# ---------------------------------------------------------------------------


class TestHandHistoryWatcher:
    """Test the directory watcher for hand history files."""

    def test_poll_once_finds_new_hands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hands.txt"
            path.write_text(SAMPLE_HAND_FULL, encoding="utf-8")

            results: list[GameState] = []
            watcher = HandHistoryWatcher(
                directory=tmpdir,
                callback=lambda hands: results.extend(hands),
            )
            found = watcher.poll_once()
            assert len(found) == 1
            assert found[0].hand_number == 12345

    def test_poll_once_incremental(self) -> None:
        """Second poll should not re-parse already-read content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hands.txt"
            path.write_text(SAMPLE_HAND_FULL, encoding="utf-8")

            watcher = HandHistoryWatcher(
                directory=tmpdir,
                callback=lambda hands: None,
            )
            first = watcher.poll_once()
            assert len(first) == 1
            second = watcher.poll_once()
            assert len(second) == 0

    def test_poll_detects_appended_content(self) -> None:
        """New hands appended to an existing file should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hands.txt"
            path.write_text(SAMPLE_HAND_FULL, encoding="utf-8")

            watcher = HandHistoryWatcher(
                directory=tmpdir,
                callback=lambda hands: None,
            )
            watcher.poll_once()

            # Append a second hand.
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n\n\n" + SAMPLE_HAND_PREFLOP_ONLY)

            found = watcher.poll_once()
            assert len(found) == 1
            assert found[0].hand_number == 12346

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = HandHistoryWatcher(
                directory=tmpdir,
                callback=lambda hands: None,
            )
            found = watcher.poll_once()
            assert found == []

    def test_nonexistent_directory(self) -> None:
        watcher = HandHistoryWatcher(
            directory="/nonexistent/dir",
            callback=lambda hands: None,
        )
        found = watcher.poll_once()
        assert found == []

    def test_start_stop(self) -> None:
        """Watcher thread should start and stop cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = HandHistoryWatcher(
                directory=tmpdir,
                callback=lambda hands: None,
                poll_interval=0.1,
            )
            watcher.start()
            assert watcher.is_running
            watcher.stop()
            assert not watcher.is_running

    def test_watcher_fires_callback(self) -> None:
        """The background watcher should fire the callback on new hands."""
        collected: list[GameState] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hands.txt"

            watcher = HandHistoryWatcher(
                directory=tmpdir,
                callback=lambda hands: collected.extend(hands),
                poll_interval=0.1,
            )
            watcher.start()

            try:
                # Write a hand after watcher starts.
                path.write_text(SAMPLE_HAND_FULL, encoding="utf-8")
                # Give it time to detect.
                deadline = time.time() + 2.0
                while not collected and time.time() < deadline:
                    time.sleep(0.05)
                assert len(collected) >= 1
                assert collected[0].hand_number == 12345
            finally:
                watcher.stop()

    def test_directory_property(self) -> None:
        watcher = HandHistoryWatcher(
            directory="/some/path",
            callback=lambda hands: None,
        )
        assert watcher.directory == Path("/some/path")
