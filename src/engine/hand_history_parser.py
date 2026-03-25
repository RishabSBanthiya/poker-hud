"""PokerStars hand history parser and directory watcher.

Parses PokerStars-format hand history text files into fully populated
``GameState`` objects.  Also provides a directory watcher that detects
new or modified hand history files and parses them incrementally.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import Callable, Optional

from src.detection.card import Card, Rank, Suit
from src.engine.game_state import (
    ActionType,
    GameState,
    Player,
    PlayerAction,
    Position,
    Street,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Card-string helpers
# ---------------------------------------------------------------------------

_RANK_MAP: dict[str, Rank] = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}

_SUIT_MAP: dict[str, Suit] = {
    "h": Suit.HEARTS,
    "d": Suit.DIAMONDS,
    "c": Suit.CLUBS,
    "s": Suit.SPADES,
}


def parse_card(text: str) -> Card:
    """Parse a two-character card string like ``Ah`` or ``Td``.

    Args:
        text: Two-character card string (rank + suit letter).

    Returns:
        A ``Card`` instance.

    Raises:
        ValueError: If the string is malformed.
    """
    text = text.strip()
    if len(text) != 2:
        raise ValueError(f"Invalid card string: {text!r}")
    rank_ch, suit_ch = text[0], text[1]
    rank = _RANK_MAP.get(rank_ch)
    suit = _SUIT_MAP.get(suit_ch)
    if rank is None or suit is None:
        raise ValueError(f"Invalid card string: {text!r}")
    return Card(rank=rank, suit=suit)


def parse_cards(text: str) -> list[Card]:
    """Parse a space-separated list of card strings.

    Args:
        text: e.g. ``"Ah Kd Js"``

    Returns:
        List of ``Card`` instances.
    """
    return [parse_card(tok) for tok in text.split() if tok]


# ---------------------------------------------------------------------------
# Regex patterns for PokerStars hand history
# ---------------------------------------------------------------------------

# Header line: hand number, game type, stakes, datetime.
_RE_HEADER = re.compile(
    r"PokerStars Hand #(?P<hand_num>\d+):\s+"
    r"(?:Hold'em No Limit|Hold'em Limit|Omaha|Tournament).*?"
    r"\(\$?(?P<sb>[\d.]+)/\$?(?P<bb>[\d.]+)(?:\s+USD)?\)"
)

# Table line: name, max seats, button seat.
_RE_TABLE = re.compile(
    r"Table '(?P<name>[^']+)'\s+(?P<max_seats>\d+)-max\s+"
    r"Seat #(?P<button>\d+) is the button"
)

# Seat line: seat number, player name, stack.
_RE_SEAT = re.compile(
    r"Seat (?P<seat>\d+): (?P<name>.+?) \(\$?(?P<stack>[\d.]+) in chips\)"
)

# Dealt-to line: hero name, hole cards.
_RE_DEALT = re.compile(
    r"Dealt to (?P<name>.+?) \[(?P<cards>[^\]]+)\]"
)

# Community card lines (FLOP / TURN / RIVER).
_RE_FLOP = re.compile(r"\*\*\* FLOP \*\*\* \[(?P<cards>[^\]]+)\]")
_RE_TURN = re.compile(r"\*\*\* TURN \*\*\* \[[^\]]+\] \[(?P<card>[^\]]+)\]")
_RE_RIVER = re.compile(r"\*\*\* RIVER \*\*\* \[[^\]]+\] \[(?P<card>[^\]]+)\]")

# Action patterns.
_RE_POST_BLIND = re.compile(
    r"(?P<name>.+?): posts (?:small|big) blind \$?(?P<amount>[\d.]+)"
)
_RE_FOLD = re.compile(r"(?P<name>.+?): folds")
_RE_CHECK = re.compile(r"(?P<name>.+?): checks")
_RE_CALL = re.compile(r"(?P<name>.+?): calls \$?(?P<amount>[\d.]+)")
_RE_BET = re.compile(r"(?P<name>.+?): bets \$?(?P<amount>[\d.]+)")
_RE_RAISE = re.compile(
    r"(?P<name>.+?): raises \$?(?P<amount>[\d.]+) to \$?(?P<total>[\d.]+)"
)
_RE_ALLIN = re.compile(
    r"(?P<name>.+?): (?:bets|calls|raises) .+ and is all-in"
)

# Summary / pot.
_RE_POT = re.compile(
    r"Total pot \$?(?P<pot>[\d.]+)(?:\s*\|\s*Rake \$?(?P<rake>[\d.]+))?"
)

# Collected.
_RE_COLLECTED = re.compile(
    r"(?P<name>.+?) collected \$?(?P<amount>[\d.]+) from (?:main )?pot"
)

# Showdown: shows cards.
_RE_SHOWS = re.compile(
    r"(?P<name>.+?): shows \[(?P<cards>[^\]]+)\]"
)

# Street section markers.
_RE_STREET_MARKER = re.compile(
    r"\*\*\* (?P<street>HOLE CARDS|FLOP|TURN|RIVER|SHOW DOWN|SUMMARY) \*\*\*"
)

# Hand separator: blank line between hands.
_HAND_SEPARATOR = re.compile(r"\n\s*\n")


# ---------------------------------------------------------------------------
# Position assignment helper
# ---------------------------------------------------------------------------

_POSITION_MAP_BY_SIZE: dict[int, list[Position]] = {
    2: [Position.SB, Position.BB],
    3: [Position.BTN, Position.SB, Position.BB],
    4: [Position.BTN, Position.SB, Position.BB, Position.UTG],
    5: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.CO],
    6: [
        Position.BTN,
        Position.SB,
        Position.BB,
        Position.UTG,
        Position.HJ,
        Position.CO,
    ],
    7: [
        Position.BTN,
        Position.SB,
        Position.BB,
        Position.UTG,
        Position.MP,
        Position.HJ,
        Position.CO,
    ],
    8: [
        Position.BTN,
        Position.SB,
        Position.BB,
        Position.UTG,
        Position.UTG1,
        Position.MP,
        Position.HJ,
        Position.CO,
    ],
    9: [
        Position.BTN,
        Position.SB,
        Position.BB,
        Position.UTG,
        Position.UTG1,
        Position.MP,
        Position.LJ,
        Position.HJ,
        Position.CO,
    ],
}


def _assign_positions(
    players: list[Player], button_seat: int
) -> None:
    """Assign positions to players based on the button seat.

    Args:
        players: List of players sorted by seat number.
        button_seat: The seat number holding the button.
    """
    n = len(players)
    if n < 2 or n > 9:
        return

    positions = _POSITION_MAP_BY_SIZE.get(n)
    if positions is None:
        return

    # Find the index of the button player.
    seat_numbers = [p.seat_number for p in players]
    try:
        btn_idx = seat_numbers.index(button_seat)
    except ValueError:
        # Button seat not found among players -- skip assignment.
        return

    # Positions are assigned starting from BTN going clockwise.
    for i, pos in enumerate(positions):
        player_idx = (btn_idx + i) % n
        players[player_idx].position = pos


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass
class ParsedHand:
    """Intermediate representation of a parsed hand before GameState.

    Attributes:
        hand_number: PokerStars hand number.
        table_name: Table name from the hand history.
        small_blind: Small blind amount.
        big_blind: Big blind amount.
        button_seat: Seat number of the button.
        players: Mapping of seat number to (name, stack) tuple.
        hero_name: Name of the hero, if ``Dealt to`` was found.
        hero_cards: Hero's hole cards.
        community_cards: All community cards dealt.
        actions: List of (street, action_type, player_name, amount).
        pot: Total pot at showdown.
        rake: Rake amount.
        winners: List of (player_name, amount) tuples.
        shown_cards: Mapping of player name to their shown cards.
    """

    hand_number: int = 0
    table_name: str = ""
    small_blind: float = 0.0
    big_blind: float = 0.0
    button_seat: int = 0
    players: dict[int, tuple[str, float]] = field(default_factory=dict)
    hero_name: str = ""
    hero_cards: list[Card] = field(default_factory=list)
    community_cards: list[Card] = field(default_factory=list)
    actions: list[tuple[Street, ActionType, str, float]] = field(
        default_factory=list
    )
    pot: float = 0.0
    rake: float = 0.0
    winners: list[tuple[str, float]] = field(default_factory=list)
    shown_cards: dict[str, list[Card]] = field(default_factory=dict)


class HandHistoryParser:
    """Parses PokerStars-format hand history text into ``GameState`` objects.

    Usage::

        parser = HandHistoryParser()
        states = parser.parse_file("/path/to/hand_history.txt")
        for gs in states:
            print(gs)
    """

    def parse_file(self, filepath: str | Path) -> list[GameState]:
        """Parse all hands in a hand history file.

        Args:
            filepath: Path to a PokerStars hand history text file.

        Returns:
            List of ``GameState`` objects, one per hand.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Hand history file not found: {path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text)

    def parse_text(self, text: str) -> list[GameState]:
        """Parse a multi-hand text block into ``GameState`` objects.

        Args:
            text: Full text content containing one or more hands.

        Returns:
            List of ``GameState`` objects, one per hand.
        """
        blocks = self._split_hands(text)
        results: list[GameState] = []
        for block in blocks:
            try:
                gs = self.parse_hand(block)
                if gs is not None:
                    results.append(gs)
            except Exception:
                logger.exception("Failed to parse hand block")
        return results

    def parse_hand(self, text_block: str) -> Optional[GameState]:
        """Parse a single hand text block into a ``GameState``.

        Args:
            text_block: Text for one complete hand.

        Returns:
            A populated ``GameState``, or ``None`` if the block could
            not be parsed (e.g. not a valid hand header).
        """
        lines = text_block.strip().splitlines()
        if not lines:
            return None

        parsed = ParsedHand()
        current_street = Street.PREFLOP

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Street markers advance current_street for action context.
            marker = _RE_STREET_MARKER.match(line)
            if marker:
                street_name = marker.group("street")
                if street_name == "FLOP":
                    current_street = Street.FLOP
                elif street_name == "TURN":
                    current_street = Street.TURN
                elif street_name == "RIVER":
                    current_street = Street.RIVER
                elif street_name == "SHOW DOWN":
                    current_street = Street.SHOWDOWN

            # Header.
            m = _RE_HEADER.match(line)
            if m:
                parsed.hand_number = int(m.group("hand_num"))
                parsed.small_blind = float(m.group("sb"))
                parsed.big_blind = float(m.group("bb"))
                continue

            # Table.
            m = _RE_TABLE.match(line)
            if m:
                parsed.table_name = m.group("name")
                parsed.button_seat = int(m.group("button"))
                continue

            # Seat.
            m = _RE_SEAT.match(line)
            if m:
                seat = int(m.group("seat"))
                name = m.group("name")
                stack = float(m.group("stack"))
                parsed.players[seat] = (name, stack)
                continue

            # Dealt to hero.
            m = _RE_DEALT.match(line)
            if m:
                parsed.hero_name = m.group("name")
                parsed.hero_cards = parse_cards(m.group("cards"))
                continue

            # Community cards.
            m = _RE_FLOP.search(line)
            if m:
                parsed.community_cards = parse_cards(m.group("cards"))
                continue

            m = _RE_TURN.search(line)
            if m:
                parsed.community_cards.append(parse_card(m.group("card")))
                continue

            m = _RE_RIVER.search(line)
            if m:
                parsed.community_cards.append(parse_card(m.group("card")))
                continue

            # Actions.
            self._parse_action(line, current_street, parsed)

            # Pot / summary.
            m = _RE_POT.search(line)
            if m:
                parsed.pot = float(m.group("pot"))
                if m.group("rake"):
                    parsed.rake = float(m.group("rake"))
                continue

            # Collected.
            m = _RE_COLLECTED.search(line)
            if m:
                parsed.winners.append(
                    (m.group("name"), float(m.group("amount")))
                )
                continue

            # Shows.
            m = _RE_SHOWS.search(line)
            if m:
                parsed.shown_cards[m.group("name")] = parse_cards(
                    m.group("cards")
                )
                continue

        if not parsed.players:
            return None

        return self._build_game_state(parsed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_hands(self, text: str) -> list[str]:
        """Split multi-hand text into individual hand blocks.

        Uses double-newline as the separator and filters out blocks
        that don't look like valid hands.

        Args:
            text: Full hand history text.

        Returns:
            List of hand text blocks.
        """
        # Split on two or more consecutive newlines.
        blocks = re.split(r"\n\s*\n\s*\n", text)
        return [b.strip() for b in blocks if "PokerStars Hand" in b]

    def _parse_action(
        self,
        line: str,
        street: Street,
        parsed: ParsedHand,
    ) -> None:
        """Try to parse an action from a single line.

        Args:
            line: A single line of hand history text.
            street: The current street for action context.
            parsed: The ``ParsedHand`` to append actions to.
        """
        # Check for all-in first (it modifies other action patterns).
        is_all_in = "and is all-in" in line

        m = _RE_POST_BLIND.match(line)
        if m:
            parsed.actions.append((
                street,
                ActionType.POST_BLIND,
                m.group("name"),
                float(m.group("amount")),
            ))
            return

        m = _RE_RAISE.match(line)
        if m:
            action_type = ActionType.ALL_IN if is_all_in else ActionType.RAISE
            parsed.actions.append((
                street,
                action_type,
                m.group("name"),
                float(m.group("total")),
            ))
            return

        m = _RE_BET.match(line)
        if m:
            action_type = ActionType.ALL_IN if is_all_in else ActionType.BET
            parsed.actions.append((
                street,
                action_type,
                m.group("name"),
                float(m.group("amount")),
            ))
            return

        m = _RE_CALL.match(line)
        if m:
            action_type = ActionType.ALL_IN if is_all_in else ActionType.CALL
            parsed.actions.append((
                street,
                action_type,
                m.group("name"),
                float(m.group("amount")),
            ))
            return

        m = _RE_CHECK.match(line)
        if m:
            parsed.actions.append((
                street,
                ActionType.CHECK,
                m.group("name"),
                0.0,
            ))
            return

        m = _RE_FOLD.match(line)
        if m:
            parsed.actions.append((
                street,
                ActionType.FOLD,
                m.group("name"),
                0.0,
            ))
            return

    def _build_game_state(self, parsed: ParsedHand) -> GameState:
        """Convert a ``ParsedHand`` into a ``GameState``.

        Args:
            parsed: The intermediate parsed representation.

        Returns:
            A fully populated ``GameState``.
        """
        # Build players sorted by seat.
        players: list[Player] = []
        hero_seat = 0
        for seat in sorted(parsed.players):
            name, stack = parsed.players[seat]
            p = Player(name=name, seat_number=seat, stack_size=stack)
            # Assign hero cards.
            if name == parsed.hero_name:
                p.hole_cards = parsed.hero_cards
                hero_seat = seat
            # Assign shown cards.
            if name in parsed.shown_cards:
                p.hole_cards = parsed.shown_cards[name]
            players.append(p)

        _assign_positions(players, parsed.button_seat)

        gs = GameState(
            table_name=parsed.table_name,
            small_blind=parsed.small_blind,
            big_blind=parsed.big_blind,
            players=players,
            hero_seat=hero_seat,
            hand_number=parsed.hand_number,
            community_cards=parsed.community_cards,
        )

        # Replay actions.
        for street, action_type, player_name, amount in parsed.actions:
            # Advance street if needed.
            while gs.current_street != street:
                try:
                    gs.advance_street()
                except ValueError:
                    break

            player = gs.get_player_by_name(player_name)
            if player is None:
                logger.warning(
                    "Action for unknown player %r in hand #%d",
                    player_name,
                    parsed.hand_number,
                )
                continue

            action = PlayerAction(
                action_type=action_type,
                amount=amount,
                street=street,
            )
            gs.record_action(player.seat_number, action)

        # Set final pot from summary if available.
        if parsed.pot > 0:
            gs.pot_size = parsed.pot

        return gs


# ---------------------------------------------------------------------------
# Directory watcher
# ---------------------------------------------------------------------------

# Default PokerStars hand history paths on macOS.
POKERSTARS_HH_DIRS: list[str] = [
    os.path.expanduser("~/Library/Application Support/PokerStars/HandHistory"),
    os.path.expanduser("~/PokerStars/HandHistory"),
]

# Callback type for when new hands are parsed.
NewHandsParsedCallback = Callable[[list[GameState]], None]


class HandHistoryWatcher:
    """Watches a directory for new or modified hand history files.

    Parses new hands incrementally (tail-like) and fires a callback
    whenever new hands are found.

    Args:
        directory: Directory to watch for hand history files.
        callback: Called with a list of new ``GameState`` objects whenever
            new hands are detected.
        poll_interval: Seconds between directory polls.
        file_pattern: Glob pattern for hand history files.
    """

    def __init__(
        self,
        directory: str | Path,
        callback: NewHandsParsedCallback,
        poll_interval: float = 1.0,
        file_pattern: str = "*.txt",
    ) -> None:
        self._directory = Path(directory)
        self._callback = callback
        self._poll_interval = poll_interval
        self._file_pattern = file_pattern
        self._parser = HandHistoryParser()
        self._file_positions: dict[Path, int] = {}
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    @property
    def directory(self) -> Path:
        """Return the watched directory path."""
        return self._directory

    @property
    def is_running(self) -> bool:
        """Return whether the watcher is actively running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start watching the directory in a background thread."""
        if self.is_running:
            logger.warning("Watcher is already running")
            return

        if not self._directory.exists():
            logger.warning(
                "Watch directory does not exist: %s", self._directory
            )

        self._stop_event.clear()
        self._thread = Thread(
            target=self._poll_loop, name="hh-watcher", daemon=True
        )
        self._thread.start()
        logger.info("Hand history watcher started: %s", self._directory)

    def stop(self) -> None:
        """Stop the watcher and wait for the background thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval * 2)
            self._thread = None
        logger.info("Hand history watcher stopped")

    def poll_once(self) -> list[GameState]:
        """Perform a single poll of the directory.

        Useful for testing or manual polling without starting the
        background thread.

        Returns:
            List of newly parsed ``GameState`` objects found in this poll.
        """
        return self._check_files()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            try:
                new_hands = self._check_files()
                if new_hands:
                    self._callback(new_hands)
            except Exception:
                logger.exception("Error in hand history poll loop")
            self._stop_event.wait(self._poll_interval)

    def _check_files(self) -> list[GameState]:
        """Check for new content in hand history files.

        Returns:
            List of newly parsed ``GameState`` objects.
        """
        all_new: list[GameState] = []

        if not self._directory.exists():
            return all_new

        for path in sorted(self._directory.glob(self._file_pattern)):
            if not path.is_file():
                continue
            try:
                new_hands = self._read_new_content(path)
                all_new.extend(new_hands)
            except Exception:
                logger.exception("Error reading %s", path)

        return all_new

    def _read_new_content(self, path: Path) -> list[GameState]:
        """Read only new content from a file (tail-like behavior).

        Args:
            path: Path to the hand history file.

        Returns:
            List of ``GameState`` objects parsed from new content.
        """
        file_size = path.stat().st_size
        last_pos = self._file_positions.get(path, 0)

        if file_size <= last_pos:
            # File hasn't grown (or was truncated).
            if file_size < last_pos:
                # File was truncated; re-read from start.
                last_pos = 0
            else:
                return []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last_pos)
            new_text = f.read()
            self._file_positions[path] = f.tell()

        if not new_text.strip():
            return []

        return self._parser.parse_text(new_text)
