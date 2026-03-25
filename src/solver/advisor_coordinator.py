"""Strategy advisor coordinator combining preflop and postflop advice.

Contains the StrategyAdvice dataclass that aggregates all solver
outputs into a single recommendation displayed on the overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.solver.postflop_advisor import ActionRecommendation


@dataclass
class DrawInfo:
    """Information about draws on the current board.

    Attributes:
        has_flush_draw: Whether hero has a flush draw.
        has_straight_draw: Whether hero has a straight draw.
        outs: Number of outs to improve.
        draw_description: Human-readable description of the draw.
    """

    has_flush_draw: bool = False
    has_straight_draw: bool = False
    outs: int = 0
    draw_description: str = ""


@dataclass
class StrategyAdvice:
    """Complete strategy advice for the current decision point.

    Attributes:
        recommendation: The primary recommended action.
        equity: Hero's hand equity as a percentage (0-100).
        pot_odds: Pot odds as a percentage (0-100).
        opponent_range_summary: Short description of estimated opponent range.
        draw_info: Information about available draws, if any.
        street: Current street (preflop/flop/turn/river).
        is_preflop: Whether this is a preflop decision.
    """

    recommendation: ActionRecommendation = field(
        default_factory=ActionRecommendation
    )
    equity: float = 0.0
    pot_odds: float = 0.0
    opponent_range_summary: str = ""
    draw_info: DrawInfo | None = None
    street: str = "preflop"
    is_preflop: bool = True
