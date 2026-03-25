"""Detection confidence scoring and validation layer.

Validates detected cards against poker rules, provides temporal consistency
tracking across frames, and computes aggregated confidence scores from
multiple detection sources.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.detection.card import Card, DetectedCard
from src.engine.game_state import STREET_COMMUNITY_CARDS, Street

logger = logging.getLogger(__name__)

# Total unique cards in a standard deck.
_DECK_SIZE = 52

# Maximum community cards on the board.
_MAX_COMMUNITY_CARDS = 5

# Hole cards per player in Texas Hold'em.
_HOLE_CARDS_PER_PLAYER = 2


class ValidationErrorType(Enum):
    """Categories of validation errors detected during card validation."""

    DUPLICATE_CARD = auto()
    INVALID_COMMUNITY_COUNT = auto()
    INVALID_HOLE_CARD_COUNT = auto()
    EXCEEDS_DECK_SIZE = auto()
    COMMUNITY_CARD_DISAPPEARED = auto()
    TEMPORAL_INCONSISTENCY = auto()
    COMMUNITY_ORDER_VIOLATION = auto()


@dataclass(frozen=True)
class ValidationError:
    """A single validation error with context.

    Attributes:
        error_type: The category of validation failure.
        message: Human-readable description of the error.
        cards: Cards involved in the error, if applicable.
    """

    error_type: ValidationErrorType
    message: str
    cards: tuple[Card, ...] = ()

    def __str__(self) -> str:
        card_str = ", ".join(str(c) for c in self.cards)
        if card_str:
            return f"[{self.error_type.name}] {self.message} ({card_str})"
        return f"[{self.error_type.name}] {self.message}"


@dataclass
class DetectionResult:
    """Wraps validated detection output with confidence and validation info.

    Attributes:
        community_cards: Cards detected on the community board.
        player_cards: Mapping of seat number to that player's hole cards.
        confidence_scores: Per-card confidence scores keyed by Card.
        validation_errors: List of validation errors found.
        is_valid: Whether the detection passed all validation checks.
        timestamp: Unix timestamp when this result was created.
    """

    community_cards: list[Card] = field(default_factory=list)
    player_cards: dict[int, list[Card]] = field(default_factory=dict)
    confidence_scores: dict[Card, float] = field(default_factory=dict)
    validation_errors: list[ValidationError] = field(default_factory=list)
    is_valid: bool = True
    timestamp: float = field(default_factory=time.time)

    def get_community_cards(self) -> list[Card]:
        """Return the community board cards.

        Returns:
            List of community cards in deal order.
        """
        return list(self.community_cards)

    def get_player_cards(self, seat: int) -> list[Card]:
        """Return hole cards for a specific player seat.

        Args:
            seat: The seat number to look up.

        Returns:
            List of hole cards for the player, empty if none detected.
        """
        return list(self.player_cards.get(seat, []))

    def get_confidence(self, card: Card) -> float:
        """Return the aggregated confidence score for a specific card.

        Args:
            card: The card to look up.

        Returns:
            Confidence score between 0.0 and 1.0, or 0.0 if not found.
        """
        return self.confidence_scores.get(card, 0.0)

    def all_cards(self) -> list[Card]:
        """Return all detected cards (community + all player hole cards).

        Returns:
            Flat list of every card in this detection result.
        """
        cards = list(self.community_cards)
        for hole_cards in self.player_cards.values():
            cards.extend(hole_cards)
        return cards


class DetectionValidator:
    """Validates detected cards against Texas Hold'em poker rules.

    Checks for duplicate cards, correct community card counts per street,
    hole card counts, deck size limits, and temporal consistency across
    multiple frames.

    Args:
        history_size: Number of past frames to retain for temporal
            consistency checks.
    """

    def __init__(self, history_size: int = 10) -> None:
        self._history_size = history_size
        self._frame_history: list[DetectionResult] = []

    @property
    def history_size(self) -> int:
        """Return the configured history buffer size."""
        return self._history_size

    @property
    def frame_history(self) -> list[DetectionResult]:
        """Return the current frame history buffer (read-only copy)."""
        return list(self._frame_history)

    def clear_history(self) -> None:
        """Clear the frame history buffer (e.g., on new hand)."""
        self._frame_history.clear()

    def validate(
        self,
        result: DetectionResult,
        street: Street = Street.PREFLOP,
        num_players: int = 0,
    ) -> DetectionResult:
        """Run all validation checks and update the result in place.

        Args:
            result: The detection result to validate.
            street: Current betting street for community card count checks.
            num_players: Number of players at the table (0 to skip hole
                card count validation against player count).

        Returns:
            The same DetectionResult with validation_errors and is_valid
            updated.
        """
        errors: list[ValidationError] = []

        errors.extend(self._check_duplicates(result))
        errors.extend(self._check_community_count(result, street))
        errors.extend(self._check_hole_card_counts(result))
        errors.extend(self._check_deck_size(result))
        errors.extend(self._check_community_order(result, street))
        errors.extend(self._check_temporal_consistency(result))

        result.validation_errors = errors
        result.is_valid = len(errors) == 0

        # Add to history after validation.
        self._frame_history.append(result)
        if len(self._frame_history) > self._history_size:
            self._frame_history.pop(0)

        if errors:
            logger.warning(
                "Detection validation found %d error(s): %s",
                len(errors),
                "; ".join(str(e) for e in errors),
            )

        return result

    def _check_duplicates(self, result: DetectionResult) -> list[ValidationError]:
        """Check that no card appears more than once across all positions."""
        errors: list[ValidationError] = []
        seen: dict[Card, str] = {}
        all_cards_with_source: list[tuple[Card, str]] = []

        for card in result.community_cards:
            all_cards_with_source.append((card, "community"))
        for seat, cards in result.player_cards.items():
            for card in cards:
                all_cards_with_source.append((card, f"seat_{seat}"))

        for card, source in all_cards_with_source:
            if card in seen:
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.DUPLICATE_CARD,
                        message=(
                            f"Card {card} appears in both {seen[card]} "
                            f"and {source}"
                        ),
                        cards=(card,),
                    )
                )
            else:
                seen[card] = source

        return errors

    def _check_community_count(
        self, result: DetectionResult, street: Street
    ) -> list[ValidationError]:
        """Check community card count is valid for the current street."""
        errors: list[ValidationError] = []
        count = len(result.community_cards)
        expected = STREET_COMMUNITY_CARDS.get(street, 0)

        if count > _MAX_COMMUNITY_CARDS:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_COMMUNITY_COUNT,
                    message=(
                        f"Detected {count} community cards, "
                        f"maximum is {_MAX_COMMUNITY_CARDS}"
                    ),
                )
            )
        elif count > expected:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.INVALID_COMMUNITY_COUNT,
                    message=(
                        f"Detected {count} community cards on "
                        f"{street.name}, expected at most {expected}"
                    ),
                )
            )

        return errors

    def _check_hole_card_counts(
        self, result: DetectionResult
    ) -> list[ValidationError]:
        """Check each player has exactly 0 or 2 hole cards."""
        errors: list[ValidationError] = []

        for seat, cards in result.player_cards.items():
            if len(cards) not in (0, _HOLE_CARDS_PER_PLAYER):
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.INVALID_HOLE_CARD_COUNT,
                        message=(
                            f"Seat {seat} has {len(cards)} hole card(s), "
                            f"expected 0 or {_HOLE_CARDS_PER_PLAYER}"
                        ),
                        cards=tuple(cards),
                    )
                )

        return errors

    def _check_deck_size(self, result: DetectionResult) -> list[ValidationError]:
        """Check total cards in play don't exceed 52."""
        errors: list[ValidationError] = []
        total = len(result.all_cards())

        if total > _DECK_SIZE:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.EXCEEDS_DECK_SIZE,
                    message=(
                        f"Total cards in play ({total}) exceeds "
                        f"deck size ({_DECK_SIZE})"
                    ),
                )
            )

        return errors

    def _check_community_order(
        self, result: DetectionResult, street: Street
    ) -> list[ValidationError]:
        """Check community cards appear in valid street progression.

        Community cards must follow: 0 (preflop) -> 3 (flop) -> 4 (turn)
        -> 5 (river). Intermediate counts like 1, 2 are invalid.
        """
        errors: list[ValidationError] = []
        count = len(result.community_cards)
        valid_counts = set(STREET_COMMUNITY_CARDS.values())

        if count > 0 and count not in valid_counts:
            errors.append(
                ValidationError(
                    error_type=ValidationErrorType.COMMUNITY_ORDER_VIOLATION,
                    message=(
                        f"Detected {count} community cards, which does not "
                        f"match any valid street count (0, 3, 4, or 5)"
                    ),
                )
            )

        return errors

    def _check_temporal_consistency(
        self, result: DetectionResult
    ) -> list[ValidationError]:
        """Check that community cards don't disappear between frames.

        Once community cards are dealt, they should persist in subsequent
        frames. A card disappearing suggests a detection error.
        """
        errors: list[ValidationError] = []

        if not self._frame_history:
            return errors

        previous = self._frame_history[-1]
        current_community = set(result.community_cards)
        previous_community = set(previous.community_cards)

        # Cards that were on the board but are now missing.
        disappeared = previous_community - current_community

        for card in disappeared:
            # Only flag if the previous frame had at least as many or more
            # community cards (a new hand would reset to 0).
            if len(result.community_cards) >= len(previous.community_cards):
                errors.append(
                    ValidationError(
                        error_type=ValidationErrorType.COMMUNITY_CARD_DISAPPEARED,
                        message=(
                            f"Community card {card} was detected in previous "
                            f"frame but is missing in current frame"
                        ),
                        cards=(card,),
                    )
                )

        return errors


@dataclass
class ConfidenceWeights:
    """Configurable weights for confidence score aggregation.

    Attributes:
        template_weight: Weight for template matching confidence.
        cnn_weight: Weight for CNN model confidence.
        temporal_weight: Weight for temporal consistency score.
        spatial_weight: Weight for spatial consistency score.
    """

    template_weight: float = 0.3
    cnn_weight: float = 0.3
    temporal_weight: float = 0.25
    spatial_weight: float = 0.15

    def __post_init__(self) -> None:
        total = (
            self.template_weight
            + self.cnn_weight
            + self.temporal_weight
            + self.spatial_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {total:.6f}"
            )


@dataclass
class ConfidenceThresholds:
    """Configurable thresholds for confidence-based decisions.

    Attributes:
        min_confidence: Minimum score to accept a detection.
        uncertain_low: Lower bound of the uncertain range.
        uncertain_high: Upper bound of the uncertain range.
    """

    min_confidence: float = 0.5
    uncertain_low: float = 0.5
    uncertain_high: float = 0.75

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )
        if self.uncertain_low > self.uncertain_high:
            raise ValueError(
                f"uncertain_low ({self.uncertain_low}) must be <= "
                f"uncertain_high ({self.uncertain_high})"
            )


class ConfidenceLevel(Enum):
    """Classification of a detection's confidence."""

    REJECTED = auto()
    UNCERTAIN = auto()
    ACCEPTED = auto()


class ConfidenceScorer:
    """Aggregates confidence from multiple detection sources.

    Combines template matching, CNN, temporal consistency, and spatial
    consistency scores into a single confidence value per card. Uses
    configurable weights and thresholds for acceptance decisions.

    Args:
        weights: Weights for each confidence source.
        thresholds: Thresholds for acceptance/rejection decisions.
        history_size: Number of past frames to retain for temporal scoring.
    """

    def __init__(
        self,
        weights: Optional[ConfidenceWeights] = None,
        thresholds: Optional[ConfidenceThresholds] = None,
        history_size: int = 10,
    ) -> None:
        self._weights = weights or ConfidenceWeights()
        self._thresholds = thresholds or ConfidenceThresholds()
        self._history_size = history_size
        self._detection_history: list[dict[Card, float]] = []

    @property
    def weights(self) -> ConfidenceWeights:
        """Return the current confidence weights."""
        return self._weights

    @property
    def thresholds(self) -> ConfidenceThresholds:
        """Return the current confidence thresholds."""
        return self._thresholds

    def clear_history(self) -> None:
        """Clear the detection history buffer (e.g., on new hand)."""
        self._detection_history.clear()

    def score(
        self,
        card: Card,
        template_confidence: float = 0.0,
        cnn_confidence: Optional[float] = None,
        spatial_confidence: float = 0.0,
    ) -> float:
        """Compute aggregated confidence score for a single card.

        Combines available confidence sources using configured weights.
        When CNN confidence is not available, its weight is redistributed
        proportionally among the other sources.

        Args:
            card: The card being scored.
            template_confidence: Confidence from template matching (0-1).
            cnn_confidence: Confidence from CNN model (0-1), or None if
                not available.
            spatial_confidence: Confidence from spatial position (0-1).

        Returns:
            Aggregated confidence score between 0.0 and 1.0.
        """
        temporal_confidence = self._compute_temporal_score(card)

        w = self._weights
        if cnn_confidence is not None:
            score = (
                w.template_weight * template_confidence
                + w.cnn_weight * cnn_confidence
                + w.temporal_weight * temporal_confidence
                + w.spatial_weight * spatial_confidence
            )
        else:
            # Redistribute CNN weight proportionally.
            remaining = w.template_weight + w.temporal_weight + w.spatial_weight
            if remaining > 0:
                scale = 1.0 / remaining
                score = (
                    w.template_weight * scale * template_confidence
                    + w.temporal_weight * scale * temporal_confidence
                    + w.spatial_weight * scale * spatial_confidence
                )
            else:
                score = 0.0

        return max(0.0, min(1.0, score))

    def score_detections(
        self,
        detections: list[DetectedCard],
        cnn_scores: Optional[dict[Card, float]] = None,
        spatial_scores: Optional[dict[Card, float]] = None,
    ) -> dict[Card, float]:
        """Score a batch of detected cards and record in history.

        Args:
            detections: List of detected cards with template confidence.
            cnn_scores: Optional CNN confidence per card.
            spatial_scores: Optional spatial confidence per card.

        Returns:
            Dictionary mapping each card to its aggregated confidence.
        """
        cnn_scores = cnn_scores or {}
        spatial_scores = spatial_scores or {}
        scores: dict[Card, float] = {}

        for det in detections:
            card = det.card
            cnn_conf = cnn_scores.get(card)
            spatial_conf = spatial_scores.get(card, 0.0)

            scores[card] = self.score(
                card=card,
                template_confidence=det.confidence,
                cnn_confidence=cnn_conf,
                spatial_confidence=spatial_conf,
            )

        # Record this frame's scores in history.
        self._detection_history.append(dict(scores))
        if len(self._detection_history) > self._history_size:
            self._detection_history.pop(0)

        return scores

    def classify(self, confidence: float) -> ConfidenceLevel:
        """Classify a confidence score into accepted/uncertain/rejected.

        Args:
            confidence: The confidence score to classify.

        Returns:
            The confidence classification level.
        """
        if confidence < self._thresholds.min_confidence:
            return ConfidenceLevel.REJECTED
        if confidence <= self._thresholds.uncertain_high:
            return ConfidenceLevel.UNCERTAIN
        return ConfidenceLevel.ACCEPTED

    def _compute_temporal_score(self, card: Card) -> float:
        """Compute temporal consistency score for a card.

        Looks at how consistently this card has been detected across
        recent frames. More consistent detection yields higher score.

        Args:
            card: The card to compute temporal score for.

        Returns:
            Temporal consistency score between 0.0 and 1.0.
        """
        if not self._detection_history:
            return 0.0

        frames_with_card = sum(
            1 for frame_scores in self._detection_history
            if card in frame_scores
        )

        return frames_with_card / len(self._detection_history)
