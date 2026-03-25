"""Postflop advisor providing action recommendations.

Contains the ActionRecommendation dataclass representing a single
recommended action with associated metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """Possible poker actions."""

    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


@dataclass
class ActionRecommendation:
    """A recommended action from the solver.

    Attributes:
        action: The recommended action.
        confidence: Confidence level 0.0-1.0.
        ev: Expected value of the action in big blinds.
        sizing: Recommended bet/raise size (fraction of pot), if applicable.
        reasoning: Human-readable explanation.
    """

    action: Action = Action.FOLD
    confidence: float = 0.0
    ev: float = 0.0
    sizing: float = 0.0
    reasoning: str = ""
