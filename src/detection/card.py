"""Card data models for the detection subsystem.

Defines enums for card suits and ranks, plus dataclasses for
representing detected cards with confidence scores and locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """Playing card suits."""

    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"

    @property
    def symbol(self) -> str:
        """Return Unicode symbol for the suit."""
        symbols = {
            Suit.HEARTS: "\u2665",
            Suit.DIAMONDS: "\u2666",
            Suit.CLUBS: "\u2663",
            Suit.SPADES: "\u2660",
        }
        return symbols[self]

    @property
    def color(self) -> tuple[int, int, int]:
        """Return BGR color tuple for the suit."""
        if self in (Suit.HEARTS, Suit.DIAMONDS):
            return (0, 0, 200)  # Red in BGR
        return (0, 0, 0)  # Black in BGR


class Rank(Enum):
    """Playing card ranks."""

    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


@dataclass(frozen=True)
class Card:
    """A playing card with suit and rank.

    Attributes:
        rank: The card's rank (2-A).
        suit: The card's suit (hearts, diamonds, clubs, spades).
    """

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.symbol}"

    @property
    def name(self) -> str:
        """Return descriptive name like 'ace_spades'."""
        rank_names = {
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.SEVEN: "7",
            Rank.EIGHT: "8",
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "jack",
            Rank.QUEEN: "queen",
            Rank.KING: "king",
            Rank.ACE: "ace",
        }
        return f"{rank_names[self.rank]}_{self.suit.value}"


@dataclass(frozen=True)
class DetectedCard:
    """A card detected in a frame with location and confidence.

    Attributes:
        rank: The card's rank.
        suit: The card's suit.
        confidence: Match confidence score (0.0 to 1.0).
        bounding_box: Location as (x, y, width, height) in pixels.
    """

    rank: Rank
    suit: Suit
    confidence: float
    bounding_box: tuple[int, int, int, int]

    @property
    def card(self) -> Card:
        """Return the underlying Card without detection metadata."""
        return Card(rank=self.rank, suit=self.suit)

    def __str__(self) -> str:
        x, y, w, h = self.bounding_box
        return (
            f"{self.rank.value}{self.suit.symbol} "
            f"(conf={self.confidence:.2f}, box=({x},{y},{w},{h}))"
        )
