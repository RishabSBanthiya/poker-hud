"""Game state validation layer for the poker engine.

Validates game state consistency after each update, checking pot sizes,
stack sizes, betting rules, and player counts. Can auto-correct minor
inconsistencies like pot rounding errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.common.logging import get_logger
from src.engine.game_state import ActionType, GameState, Street

logger = get_logger("engine.state_validator")

# Tolerance for floating-point comparison of chip amounts.
_CHIP_TOLERANCE = 0.01

# Maximum pot-to-sum-of-bets discrepancy that can be auto-corrected.
_AUTO_CORRECT_THRESHOLD = 1.0


class Severity(Enum):
    """Severity level for a validation error."""

    WARNING = auto()
    ERROR = auto()


@dataclass(frozen=True)
class ValidationError:
    """A single game state validation error.

    Attributes:
        severity: How severe the issue is.
        message: Human-readable description of the problem.
        context: Additional structured data for debugging.
    """

    severity: Severity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        prefix = self.severity.name
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{prefix}] {self.message} ({ctx})"
        return f"[{prefix}] {self.message}"


class GameStateValidator:
    """Validates game state consistency after each update.

    Checks include:
    - Pot size equals sum of all bets
    - Stack sizes are non-negative
    - Active player count is consistent
    - Betting amounts follow poker rules (min raise = previous raise)
    - Blind structure is consistent

    Minor inconsistencies (e.g., pot rounding errors under a threshold)
    are auto-corrected in place when ``auto_correct`` is enabled.

    Args:
        auto_correct: If True, fix minor inconsistencies in place.
    """

    def __init__(self, auto_correct: bool = True) -> None:
        self._auto_correct = auto_correct

    def validate(self, state: GameState) -> list[ValidationError]:
        """Run all validation checks on the given game state.

        Args:
            state: The game state to validate.

        Returns:
            List of validation errors found. Empty list means the state
            is fully consistent.
        """
        errors: list[ValidationError] = []

        errors.extend(self._check_pot_consistency(state))
        errors.extend(self._check_stack_sizes(state))
        errors.extend(self._check_active_player_count(state))
        errors.extend(self._check_betting_rules(state))
        errors.extend(self._check_blind_structure(state))
        errors.extend(self._check_community_card_count(state))

        if errors:
            warnings = [e for e in errors if e.severity == Severity.WARNING]
            hard_errors = [e for e in errors if e.severity == Severity.ERROR]
            if hard_errors:
                logger.warning(
                    "state_validation_errors",
                    error_count=len(hard_errors),
                    warning_count=len(warnings),
                    errors=[str(e) for e in hard_errors],
                )
            elif warnings:
                logger.debug(
                    "state_validation_warnings",
                    warning_count=len(warnings),
                )

        return errors

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_pot_consistency(
        self, state: GameState
    ) -> list[ValidationError]:
        """Check that pot size equals the sum of all bets in action history."""
        errors: list[ValidationError] = []

        bet_actions = {
            ActionType.BET,
            ActionType.RAISE,
            ActionType.CALL,
            ActionType.ALL_IN,
            ActionType.POST_BLIND,
        }
        total_bets = sum(
            a.amount for a in state.action_history if a.action_type in bet_actions
        )

        diff = abs(state.pot_size - total_bets)
        if diff > _CHIP_TOLERANCE:
            if diff <= _AUTO_CORRECT_THRESHOLD and self._auto_correct:
                old_pot = state.pot_size
                state.pot_size = total_bets
                errors.append(
                    ValidationError(
                        severity=Severity.WARNING,
                        message=(
                            f"Pot size auto-corrected from {old_pot:.2f} "
                            f"to {total_bets:.2f}"
                        ),
                        context={
                            "old_pot": old_pot,
                            "corrected_pot": total_bets,
                            "diff": diff,
                        },
                    )
                )
            else:
                errors.append(
                    ValidationError(
                        severity=Severity.ERROR,
                        message=(
                            f"Pot size ({state.pot_size:.2f}) does not match "
                            f"sum of bets ({total_bets:.2f})"
                        ),
                        context={
                            "pot_size": state.pot_size,
                            "sum_of_bets": total_bets,
                            "diff": diff,
                        },
                    )
                )

        return errors

    def _check_stack_sizes(
        self, state: GameState
    ) -> list[ValidationError]:
        """Check that all player stack sizes are non-negative."""
        errors: list[ValidationError] = []

        for player in state.players:
            if player.stack_size < -_CHIP_TOLERANCE:
                errors.append(
                    ValidationError(
                        severity=Severity.ERROR,
                        message=(
                            f"Player '{player.name}' (seat {player.seat_number}) "
                            f"has negative stack: {player.stack_size:.2f}"
                        ),
                        context={
                            "player_name": player.name,
                            "seat": player.seat_number,
                            "stack_size": player.stack_size,
                        },
                    )
                )

        return errors

    def _check_active_player_count(
        self, state: GameState
    ) -> list[ValidationError]:
        """Check active player count consistency."""
        errors: list[ValidationError] = []

        if not state.players:
            return errors

        active_count = len(state.get_active_players())
        total_count = state.get_num_players()

        # Count folds in action history
        fold_count = sum(
            1
            for a in state.action_history
            if a.action_type == ActionType.FOLD
        )

        expected_active = total_count - fold_count
        if active_count != expected_active:
            errors.append(
                ValidationError(
                    severity=Severity.WARNING,
                    message=(
                        f"Active player count ({active_count}) does not match "
                        f"expected ({expected_active} = {total_count} total "
                        f"- {fold_count} folds)"
                    ),
                    context={
                        "active_count": active_count,
                        "expected_active": expected_active,
                        "total_players": total_count,
                        "fold_count": fold_count,
                    },
                )
            )

        # There should be at least 1 active player during a hand
        if active_count == 0 and total_count > 0 and state.action_history:
            errors.append(
                ValidationError(
                    severity=Severity.ERROR,
                    message="No active players remaining but hand has actions",
                    context={"total_players": total_count},
                )
            )

        return errors

    def _check_betting_rules(
        self, state: GameState
    ) -> list[ValidationError]:
        """Check that betting amounts follow poker rules.

        Validates min-raise amounts: a raise must be at least as large as
        the previous raise increment.
        """
        errors: list[ValidationError] = []

        if not state.action_history:
            return errors

        # Group actions by street
        actions_by_street: dict[Street, list[tuple[str, ActionType, float]]] = {}
        for action in state.action_history:
            street = action.street
            if street not in actions_by_street:
                actions_by_street[street] = []
            actions_by_street[street].append(
                (action.player_name, action.action_type, action.amount)
            )

        for street, actions in actions_by_street.items():
            last_bet_amount = 0.0
            last_raise_increment = 0.0

            # For preflop, the BB is the initial "bet"
            if street == Street.PREFLOP and state.big_blind > 0:
                last_bet_amount = state.big_blind
                last_raise_increment = state.big_blind

            for player_name, action_type, amount in actions:
                if action_type == ActionType.POST_BLIND:
                    continue

                if action_type == ActionType.RAISE:
                    raise_total = amount
                    # The raise increment is the amount above the previous bet
                    if raise_total < last_raise_increment - _CHIP_TOLERANCE:
                        errors.append(
                            ValidationError(
                                severity=Severity.WARNING,
                                message=(
                                    f"Player '{player_name}' raise of "
                                    f"{raise_total:.2f} is below min raise "
                                    f"of {last_raise_increment:.2f} "
                                    f"on {street.name}"
                                ),
                                context={
                                    "player_name": player_name,
                                    "raise_amount": raise_total,
                                    "min_raise": last_raise_increment,
                                    "street": street.name,
                                },
                            )
                        )
                    last_raise_increment = max(raise_total, last_raise_increment)
                    last_bet_amount += raise_total

                elif action_type == ActionType.BET:
                    if amount < state.big_blind - _CHIP_TOLERANCE:
                        errors.append(
                            ValidationError(
                                severity=Severity.WARNING,
                                message=(
                                    f"Player '{player_name}' bet of "
                                    f"{amount:.2f} is below big blind "
                                    f"({state.big_blind:.2f}) on {street.name}"
                                ),
                                context={
                                    "player_name": player_name,
                                    "bet_amount": amount,
                                    "big_blind": state.big_blind,
                                    "street": street.name,
                                },
                            )
                        )
                    last_bet_amount = amount
                    last_raise_increment = amount

        return errors

    def _check_blind_structure(
        self, state: GameState
    ) -> list[ValidationError]:
        """Check that blind structure is consistent."""
        errors: list[ValidationError] = []

        if state.small_blind < 0:
            errors.append(
                ValidationError(
                    severity=Severity.ERROR,
                    message=f"Small blind is negative: {state.small_blind:.2f}",
                    context={"small_blind": state.small_blind},
                )
            )

        if state.big_blind < 0:
            errors.append(
                ValidationError(
                    severity=Severity.ERROR,
                    message=f"Big blind is negative: {state.big_blind:.2f}",
                    context={"big_blind": state.big_blind},
                )
            )

        if (
            state.small_blind > 0
            and state.big_blind > 0
            and state.small_blind > state.big_blind + _CHIP_TOLERANCE
        ):
            errors.append(
                ValidationError(
                    severity=Severity.ERROR,
                    message=(
                        f"Small blind ({state.small_blind:.2f}) is greater "
                        f"than big blind ({state.big_blind:.2f})"
                    ),
                    context={
                        "small_blind": state.small_blind,
                        "big_blind": state.big_blind,
                    },
                )
            )

        return errors

    def _check_community_card_count(
        self, state: GameState
    ) -> list[ValidationError]:
        """Check community card count is valid for the current street."""
        errors: list[ValidationError] = []

        from src.engine.game_state import STREET_COMMUNITY_CARDS

        expected = STREET_COMMUNITY_CARDS.get(state.current_street, 0)
        actual = len(state.community_cards)

        if actual > expected:
            errors.append(
                ValidationError(
                    severity=Severity.ERROR,
                    message=(
                        f"Too many community cards ({actual}) for "
                        f"{state.current_street.name} (expected at most {expected})"
                    ),
                    context={
                        "actual_count": actual,
                        "expected_max": expected,
                        "street": state.current_street.name,
                    },
                )
            )

        return errors
