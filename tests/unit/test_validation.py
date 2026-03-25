"""Unit tests for src.detection.validation module."""

from __future__ import annotations

import pytest
from src.detection.card import Card, DetectedCard, Rank, Suit
from src.detection.validation import (
    ConfidenceLevel,
    ConfidenceScorer,
    ConfidenceThresholds,
    ConfidenceWeights,
    DetectionResult,
    DetectionValidator,
    ValidationError,
    ValidationErrorType,
)
from src.engine.game_state import Street

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _card(rank: Rank, suit: Suit) -> Card:
    return Card(rank=rank, suit=suit)


def _detected(rank: Rank, suit: Suit, confidence: float = 0.9) -> DetectedCard:
    return DetectedCard(
        rank=rank, suit=suit, confidence=confidence, bounding_box=(0, 0, 60, 80)
    )


AS = _card(Rank.ACE, Suit.SPADES)
KH = _card(Rank.KING, Suit.HEARTS)
QD = _card(Rank.QUEEN, Suit.DIAMONDS)
JC = _card(Rank.JACK, Suit.CLUBS)
TS = _card(Rank.TEN, Suit.SPADES)
NH = _card(Rank.NINE, Suit.HEARTS)
ED = _card(Rank.EIGHT, Suit.DIAMONDS)
SC = _card(Rank.SEVEN, Suit.CLUBS)
FH = _card(Rank.FIVE, Suit.HEARTS)
TD = _card(Rank.TWO, Suit.DIAMONDS)


# -----------------------------------------------------------------------
# ValidationError tests
# -----------------------------------------------------------------------


class TestValidationError:
    def test_str_with_cards(self) -> None:
        err = ValidationError(
            error_type=ValidationErrorType.DUPLICATE_CARD,
            message="duplicate found",
            cards=(AS,),
        )
        result = str(err)
        assert "DUPLICATE_CARD" in result
        assert "duplicate found" in result

    def test_str_without_cards(self) -> None:
        err = ValidationError(
            error_type=ValidationErrorType.EXCEEDS_DECK_SIZE,
            message="too many cards",
        )
        result = str(err)
        assert "EXCEEDS_DECK_SIZE" in result
        assert "too many cards" in result


# -----------------------------------------------------------------------
# DetectionResult tests
# -----------------------------------------------------------------------


class TestDetectionResult:
    def test_defaults(self) -> None:
        result = DetectionResult()
        assert result.community_cards == []
        assert result.player_cards == {}
        assert result.confidence_scores == {}
        assert result.validation_errors == []
        assert result.is_valid is True
        assert result.timestamp > 0

    def test_get_community_cards(self) -> None:
        result = DetectionResult(community_cards=[AS, KH, QD])
        cards = result.get_community_cards()
        assert cards == [AS, KH, QD]
        # Returns a copy.
        cards.append(JC)
        assert len(result.community_cards) == 3

    def test_get_player_cards(self) -> None:
        result = DetectionResult(player_cards={0: [AS, KH], 1: [QD, JC]})
        assert result.get_player_cards(0) == [AS, KH]
        assert result.get_player_cards(1) == [QD, JC]
        # Missing seat returns empty list.
        assert result.get_player_cards(5) == []

    def test_get_player_cards_returns_copy(self) -> None:
        result = DetectionResult(player_cards={0: [AS, KH]})
        cards = result.get_player_cards(0)
        cards.append(QD)
        assert len(result.player_cards[0]) == 2

    def test_get_confidence(self) -> None:
        result = DetectionResult(confidence_scores={AS: 0.95, KH: 0.80})
        assert result.get_confidence(AS) == 0.95
        assert result.get_confidence(KH) == 0.80
        # Unknown card returns 0.0.
        assert result.get_confidence(QD) == 0.0

    def test_all_cards(self) -> None:
        result = DetectionResult(
            community_cards=[AS, KH, QD],
            player_cards={0: [JC, TS], 1: [NH, ED]},
        )
        all_cards = result.all_cards()
        assert len(all_cards) == 7
        assert AS in all_cards
        assert JC in all_cards
        assert NH in all_cards


# -----------------------------------------------------------------------
# DetectionValidator — duplicate checks
# -----------------------------------------------------------------------


class TestValidatorDuplicates:
    def test_no_duplicates_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(
            community_cards=[AS, KH, QD],
            player_cards={0: [JC, TS]},
        )
        validated = validator.validate(result, street=Street.FLOP)
        assert validated.is_valid

    def test_duplicate_in_community(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, AS])
        validated = validator.validate(result, street=Street.FLOP)
        assert not validated.is_valid
        dup_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.DUPLICATE_CARD
        ]
        assert len(dup_errors) == 1
        assert AS in dup_errors[0].cards

    def test_duplicate_between_community_and_hole(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(
            community_cards=[AS, KH, QD],
            player_cards={0: [AS, JC]},
        )
        validated = validator.validate(result, street=Street.FLOP)
        assert not validated.is_valid
        dup_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.DUPLICATE_CARD
        ]
        assert len(dup_errors) == 1

    def test_duplicate_between_players(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(
            player_cards={0: [AS, KH], 1: [AS, QD]},
        )
        validated = validator.validate(result, street=Street.PREFLOP)
        dup_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.DUPLICATE_CARD
        ]
        assert len(dup_errors) == 1


# -----------------------------------------------------------------------
# DetectionValidator — community card count
# -----------------------------------------------------------------------


class TestValidatorCommunityCount:
    def test_preflop_no_cards_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[])
        validated = validator.validate(result, street=Street.PREFLOP)
        assert validated.is_valid

    def test_preflop_with_cards_invalid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, QD])
        validated = validator.validate(result, street=Street.PREFLOP)
        count_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_COMMUNITY_COUNT
        ]
        assert len(count_errors) == 1

    def test_flop_three_cards_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, QD])
        validated = validator.validate(result, street=Street.FLOP)
        count_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_COMMUNITY_COUNT
        ]
        assert len(count_errors) == 0

    def test_flop_four_cards_invalid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, QD, JC])
        validated = validator.validate(result, street=Street.FLOP)
        count_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_COMMUNITY_COUNT
        ]
        assert len(count_errors) == 1

    def test_turn_four_cards_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, QD, JC])
        validated = validator.validate(result, street=Street.TURN)
        count_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_COMMUNITY_COUNT
        ]
        assert len(count_errors) == 0

    def test_river_five_cards_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, QD, JC, TS])
        validated = validator.validate(result, street=Street.RIVER)
        count_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_COMMUNITY_COUNT
        ]
        assert len(count_errors) == 0

    def test_more_than_five_community_cards(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(
            community_cards=[AS, KH, QD, JC, TS, NH]
        )
        validated = validator.validate(result, street=Street.RIVER)
        count_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_COMMUNITY_COUNT
        ]
        assert len(count_errors) == 1


# -----------------------------------------------------------------------
# DetectionValidator — hole card counts
# -----------------------------------------------------------------------


class TestValidatorHoleCards:
    def test_two_hole_cards_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(player_cards={0: [AS, KH]})
        validated = validator.validate(result, street=Street.PREFLOP)
        hole_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_HOLE_CARD_COUNT
        ]
        assert len(hole_errors) == 0

    def test_zero_hole_cards_valid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(player_cards={0: []})
        validated = validator.validate(result, street=Street.PREFLOP)
        hole_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_HOLE_CARD_COUNT
        ]
        assert len(hole_errors) == 0

    def test_one_hole_card_invalid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(player_cards={0: [AS]})
        validated = validator.validate(result, street=Street.PREFLOP)
        hole_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_HOLE_CARD_COUNT
        ]
        assert len(hole_errors) == 1

    def test_three_hole_cards_invalid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(player_cards={0: [AS, KH, QD]})
        validated = validator.validate(result, street=Street.PREFLOP)
        hole_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_HOLE_CARD_COUNT
        ]
        assert len(hole_errors) == 1

    def test_multiple_players_mixed_validity(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(
            player_cards={0: [AS, KH], 1: [QD], 2: [JC, TS]},
        )
        validated = validator.validate(result, street=Street.PREFLOP)
        hole_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.INVALID_HOLE_CARD_COUNT
        ]
        # Only seat 1 is invalid.
        assert len(hole_errors) == 1
        assert "Seat 1" in hole_errors[0].message


# -----------------------------------------------------------------------
# DetectionValidator — deck size
# -----------------------------------------------------------------------


class TestValidatorDeckSize:
    def test_within_deck_size(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(
            community_cards=[AS, KH, QD],
            player_cards={0: [JC, TS]},
        )
        validated = validator.validate(result, street=Street.FLOP)
        deck_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.EXCEEDS_DECK_SIZE
        ]
        assert len(deck_errors) == 0

    def test_exceeds_deck_size(self) -> None:
        """Create a scenario with >52 cards to trigger deck size error."""
        validator = DetectionValidator()
        # Build 27 unique player hands (54 cards) — exceeds 52.
        player_cards: dict[int, list[Card]] = {}
        ranks = list(Rank)
        suits = list(Suit)
        card_idx = 0
        for seat in range(27):
            cards: list[Card] = []
            for _ in range(2):
                r = ranks[card_idx % len(ranks)]
                s = suits[card_idx % len(suits)]
                cards.append(Card(rank=r, suit=s))
                card_idx += 1
            player_cards[seat] = cards

        result = DetectionResult(player_cards=player_cards)
        validated = validator.validate(result, street=Street.PREFLOP)
        deck_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.EXCEEDS_DECK_SIZE
        ]
        assert len(deck_errors) == 1


# -----------------------------------------------------------------------
# DetectionValidator — community card order
# -----------------------------------------------------------------------


class TestValidatorCommunityOrder:
    def test_valid_counts(self) -> None:
        """Counts 0, 3, 4, 5 are all valid community card counts."""
        validator = DetectionValidator()
        for count, street in [
            (0, Street.PREFLOP),
            (3, Street.FLOP),
            (4, Street.TURN),
            (5, Street.RIVER),
        ]:
            cards = [AS, KH, QD, JC, TS][:count]
            result = DetectionResult(community_cards=cards)
            validated = validator.validate(result, street=street)
            order_errors = [
                e for e in validated.validation_errors
                if e.error_type == ValidationErrorType.COMMUNITY_ORDER_VIOLATION
            ]
            assert len(order_errors) == 0, f"count={count} should be valid"

    def test_one_community_card_invalid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS])
        validated = validator.validate(result, street=Street.FLOP)
        order_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.COMMUNITY_ORDER_VIOLATION
        ]
        assert len(order_errors) == 1

    def test_two_community_cards_invalid(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH])
        validated = validator.validate(result, street=Street.FLOP)
        order_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.COMMUNITY_ORDER_VIOLATION
        ]
        assert len(order_errors) == 1


# -----------------------------------------------------------------------
# DetectionValidator — temporal consistency
# -----------------------------------------------------------------------


class TestValidatorTemporalConsistency:
    def test_first_frame_no_errors(self) -> None:
        validator = DetectionValidator()
        result = DetectionResult(community_cards=[AS, KH, QD])
        validated = validator.validate(result, street=Street.FLOP)
        temporal_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.COMMUNITY_CARD_DISAPPEARED
        ]
        assert len(temporal_errors) == 0

    def test_cards_persist_no_error(self) -> None:
        validator = DetectionValidator()
        # Frame 1: flop.
        r1 = DetectionResult(community_cards=[AS, KH, QD])
        validator.validate(r1, street=Street.FLOP)
        # Frame 2: same flop + turn card.
        r2 = DetectionResult(community_cards=[AS, KH, QD, JC])
        validated = validator.validate(r2, street=Street.TURN)
        temporal_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.COMMUNITY_CARD_DISAPPEARED
        ]
        assert len(temporal_errors) == 0

    def test_card_disappears_flagged(self) -> None:
        validator = DetectionValidator()
        # Frame 1: flop.
        r1 = DetectionResult(community_cards=[AS, KH, QD])
        validator.validate(r1, street=Street.FLOP)
        # Frame 2: one card missing but same count (detection error).
        r2 = DetectionResult(community_cards=[AS, KH, JC])
        validated = validator.validate(r2, street=Street.FLOP)
        temporal_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.COMMUNITY_CARD_DISAPPEARED
        ]
        assert len(temporal_errors) == 1
        assert QD in temporal_errors[0].cards

    def test_new_hand_reset_no_false_positive(self) -> None:
        """Going from 5 community cards to 0 (new hand) should not flag."""
        validator = DetectionValidator()
        r1 = DetectionResult(community_cards=[AS, KH, QD, JC, TS])
        validator.validate(r1, street=Street.RIVER)
        # New hand: 0 community cards (fewer than previous).
        r2 = DetectionResult(community_cards=[])
        validated = validator.validate(r2, street=Street.PREFLOP)
        temporal_errors = [
            e for e in validated.validation_errors
            if e.error_type == ValidationErrorType.COMMUNITY_CARD_DISAPPEARED
        ]
        assert len(temporal_errors) == 0

    def test_clear_history(self) -> None:
        validator = DetectionValidator()
        r1 = DetectionResult(community_cards=[AS, KH, QD])
        validator.validate(r1, street=Street.FLOP)
        assert len(validator.frame_history) == 1
        validator.clear_history()
        assert len(validator.frame_history) == 0

    def test_history_size_limit(self) -> None:
        validator = DetectionValidator(history_size=3)
        for _ in range(5):
            result = DetectionResult(community_cards=[AS, KH, QD])
            validator.validate(result, street=Street.FLOP)
        assert len(validator.frame_history) == 3


# -----------------------------------------------------------------------
# ConfidenceWeights tests
# -----------------------------------------------------------------------


class TestConfidenceWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = ConfidenceWeights()
        total = w.template_weight + w.cnn_weight + w.temporal_weight + w.spatial_weight
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_valid(self) -> None:
        w = ConfidenceWeights(
            template_weight=0.4,
            cnn_weight=0.3,
            temporal_weight=0.2,
            spatial_weight=0.1,
        )
        assert w.template_weight == 0.4

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ConfidenceWeights(
                template_weight=0.5,
                cnn_weight=0.5,
                temporal_weight=0.5,
                spatial_weight=0.5,
            )


# -----------------------------------------------------------------------
# ConfidenceThresholds tests
# -----------------------------------------------------------------------


class TestConfidenceThresholds:
    def test_default_thresholds(self) -> None:
        t = ConfidenceThresholds()
        assert t.min_confidence == 0.5
        assert t.uncertain_low == 0.5
        assert t.uncertain_high == 0.75

    def test_invalid_min_confidence(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            ConfidenceThresholds(min_confidence=1.5)

    def test_uncertain_range_inverted(self) -> None:
        with pytest.raises(ValueError, match="uncertain_low"):
            ConfidenceThresholds(uncertain_low=0.8, uncertain_high=0.5)


# -----------------------------------------------------------------------
# ConfidenceScorer tests
# -----------------------------------------------------------------------


class TestConfidenceScorerScore:
    def test_all_sources_available(self) -> None:
        scorer = ConfidenceScorer()
        score = scorer.score(
            card=AS,
            template_confidence=0.9,
            cnn_confidence=0.85,
            spatial_confidence=0.8,
        )
        # With default weights: 0.3*0.9 + 0.3*0.85 + 0.25*0 + 0.15*0.8
        # temporal = 0.0 (no history)
        # = 0.27 + 0.255 + 0.0 + 0.12 = 0.645
        assert 0.64 < score < 0.66

    def test_no_cnn_redistributes_weight(self) -> None:
        scorer = ConfidenceScorer()
        score = scorer.score(
            card=AS,
            template_confidence=1.0,
            cnn_confidence=None,
            spatial_confidence=1.0,
        )
        # Without CNN, weights are redistributed among template, temporal, spatial.
        # remaining = 0.3 + 0.25 + 0.15 = 0.7; scale = 1/0.7 ~ 1.4286
        # template: 0.3 * 1.4286 * 1.0 = 0.4286
        # temporal: 0.25 * 1.4286 * 0.0 = 0.0 (no history)
        # spatial:  0.15 * 1.4286 * 1.0 = 0.2143
        # total = 0.6429
        assert 0.64 < score < 0.65

    def test_score_clamped_to_0_1(self) -> None:
        scorer = ConfidenceScorer()
        score = scorer.score(
            card=AS,
            template_confidence=0.0,
            cnn_confidence=0.0,
            spatial_confidence=0.0,
        )
        assert score == 0.0

    def test_perfect_scores(self) -> None:
        scorer = ConfidenceScorer()
        # Build history so temporal score is 1.0.
        for _ in range(5):
            scorer.score_detections([_detected(Rank.ACE, Suit.SPADES, 1.0)])
        score = scorer.score(
            card=AS,
            template_confidence=1.0,
            cnn_confidence=1.0,
            spatial_confidence=1.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)


class TestConfidenceScorerBatch:
    def test_score_detections(self) -> None:
        scorer = ConfidenceScorer()
        detections = [
            _detected(Rank.ACE, Suit.SPADES, 0.9),
            _detected(Rank.KING, Suit.HEARTS, 0.8),
        ]
        scores = scorer.score_detections(detections)
        assert AS in scores
        assert KH in scores
        assert 0.0 <= scores[AS] <= 1.0
        assert 0.0 <= scores[KH] <= 1.0

    def test_score_detections_with_cnn(self) -> None:
        scorer = ConfidenceScorer()
        detections = [_detected(Rank.ACE, Suit.SPADES, 0.9)]
        cnn_scores = {AS: 0.95}
        scores = scorer.score_detections(detections, cnn_scores=cnn_scores)
        # CNN available, so all 4 sources used.
        assert AS in scores

    def test_history_builds_up(self) -> None:
        scorer = ConfidenceScorer()
        det = [_detected(Rank.ACE, Suit.SPADES, 0.9)]
        scorer.score_detections(det)
        scorer.score_detections(det)
        assert len(scorer._detection_history) == 2

    def test_history_limit(self) -> None:
        scorer = ConfidenceScorer(history_size=3)
        det = [_detected(Rank.ACE, Suit.SPADES, 0.9)]
        for _ in range(5):
            scorer.score_detections(det)
        assert len(scorer._detection_history) == 3


class TestConfidenceScorerClassify:
    def test_rejected(self) -> None:
        scorer = ConfidenceScorer()
        assert scorer.classify(0.3) == ConfidenceLevel.REJECTED

    def test_uncertain(self) -> None:
        scorer = ConfidenceScorer()
        assert scorer.classify(0.6) == ConfidenceLevel.UNCERTAIN

    def test_accepted(self) -> None:
        scorer = ConfidenceScorer()
        assert scorer.classify(0.9) == ConfidenceLevel.ACCEPTED

    def test_boundary_min_confidence(self) -> None:
        scorer = ConfidenceScorer()
        # Exactly at min_confidence (0.5) should be uncertain, not rejected.
        assert scorer.classify(0.5) == ConfidenceLevel.UNCERTAIN

    def test_boundary_uncertain_high(self) -> None:
        scorer = ConfidenceScorer()
        # Exactly at uncertain_high (0.75) should still be uncertain.
        assert scorer.classify(0.75) == ConfidenceLevel.UNCERTAIN

    def test_just_above_uncertain_high(self) -> None:
        scorer = ConfidenceScorer()
        assert scorer.classify(0.76) == ConfidenceLevel.ACCEPTED

    def test_custom_thresholds(self) -> None:
        thresholds = ConfidenceThresholds(
            min_confidence=0.3, uncertain_low=0.3, uncertain_high=0.6
        )
        scorer = ConfidenceScorer(thresholds=thresholds)
        assert scorer.classify(0.2) == ConfidenceLevel.REJECTED
        assert scorer.classify(0.4) == ConfidenceLevel.UNCERTAIN
        assert scorer.classify(0.7) == ConfidenceLevel.ACCEPTED


class TestConfidenceScorerTemporal:
    def test_temporal_score_increases_with_consistency(self) -> None:
        scorer = ConfidenceScorer()
        det = [_detected(Rank.ACE, Suit.SPADES, 0.9)]
        # Build up history with AS present.
        for _ in range(5):
            scorer.score_detections(det)
        # Now score with full history — temporal should be high.
        score_with_history = scorer.score(
            card=AS,
            template_confidence=0.9,
            cnn_confidence=0.9,
            spatial_confidence=0.9,
        )
        # Compare to scorer with no history.
        fresh_scorer = ConfidenceScorer()
        score_no_history = fresh_scorer.score(
            card=AS,
            template_confidence=0.9,
            cnn_confidence=0.9,
            spatial_confidence=0.9,
        )
        assert score_with_history > score_no_history

    def test_temporal_score_zero_for_new_card(self) -> None:
        scorer = ConfidenceScorer()
        # History has only KH, not AS.
        det = [_detected(Rank.KING, Suit.HEARTS, 0.9)]
        for _ in range(5):
            scorer.score_detections(det)
        # Temporal score for AS should be 0.
        temporal = scorer._compute_temporal_score(AS)
        assert temporal == 0.0

    def test_clear_history(self) -> None:
        scorer = ConfidenceScorer()
        det = [_detected(Rank.ACE, Suit.SPADES, 0.9)]
        scorer.score_detections(det)
        scorer.clear_history()
        assert len(scorer._detection_history) == 0


# -----------------------------------------------------------------------
# Integration-style: validator + scorer together
# -----------------------------------------------------------------------


class TestValidatorScorerIntegration:
    def test_valid_flop_with_scores(self) -> None:
        validator = DetectionValidator()
        scorer = ConfidenceScorer()

        detections = [
            _detected(Rank.ACE, Suit.SPADES, 0.9),
            _detected(Rank.KING, Suit.HEARTS, 0.85),
            _detected(Rank.QUEEN, Suit.DIAMONDS, 0.88),
            _detected(Rank.JACK, Suit.CLUBS, 0.92),
            _detected(Rank.TEN, Suit.SPADES, 0.87),
        ]

        scores = scorer.score_detections(detections)
        result = DetectionResult(
            community_cards=[AS, KH, QD],
            player_cards={0: [JC, TS]},
            confidence_scores=scores,
        )
        validated = validator.validate(result, street=Street.FLOP)
        assert validated.is_valid
        assert validated.get_confidence(AS) > 0.0

    def test_invalid_detection_still_scored(self) -> None:
        """Even invalid detections get confidence scores."""
        validator = DetectionValidator()
        scorer = ConfidenceScorer()

        detections = [
            _detected(Rank.ACE, Suit.SPADES, 0.9),
            _detected(Rank.ACE, Suit.SPADES, 0.85),  # duplicate
        ]
        scores = scorer.score_detections(detections)

        result = DetectionResult(
            community_cards=[AS, AS, QD],  # duplicate
            confidence_scores=scores,
        )
        validated = validator.validate(result, street=Street.FLOP)
        assert not validated.is_valid
        # Scores still present.
        assert AS in validated.confidence_scores
