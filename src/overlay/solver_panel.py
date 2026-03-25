"""Strategy advisor display panel for the overlay.

Displays GTO solver recommendations including recommended action,
equity, opponent range estimation, and draw information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from src.solver.advisor_coordinator import (
    ActionRecommendation,
    DrawInfo,
    StrategyAdvice,
)

logger = logging.getLogger(__name__)


class ActionColor(Enum):
    """Color categories for action recommendations.

    Values are RGBA tuples (0.0-1.0).
    """

    GREEN = (0.0, 0.8, 0.0, 1.0)   # Raise / bet
    YELLOW = (1.0, 0.8, 0.0, 1.0)  # Call / check
    RED = (1.0, 0.2, 0.2, 1.0)     # Fold
    WHITE = (1.0, 1.0, 1.0, 1.0)   # Default / no advice


# Map actions to display colors
_ACTION_COLORS: dict[ActionRecommendation, ActionColor] = {
    ActionRecommendation.RAISE: ActionColor.GREEN,
    ActionRecommendation.ALL_IN: ActionColor.GREEN,
    ActionRecommendation.CALL: ActionColor.YELLOW,
    ActionRecommendation.CHECK: ActionColor.YELLOW,
    ActionRecommendation.FOLD: ActionColor.RED,
}


def get_action_color(action: ActionRecommendation) -> ActionColor:
    """Get the display color for an action.

    Args:
        action: The poker action recommendation.

    Returns:
        Color category for the action.
    """
    return _ACTION_COLORS.get(action, ActionColor.WHITE)


def format_action(action: ActionRecommendation) -> str:
    """Format an action for display.

    Args:
        action: The poker action recommendation.

    Returns:
        Human-readable action string.
    """
    return action.name.upper()


@dataclass
class SolverDisplayState:
    """Current display state of the solver panel.

    Attributes:
        advice: The current strategy advice, if any.
        expanded: Whether the panel is in expanded view.
        action_text: Formatted action recommendation text.
        equity_text: Formatted equity display text.
        sizing_text: Formatted recommended sizing text.
        reasoning_text: Combined reasoning text.
        draw_text: Draw information text.
        action_color: Color for the action display.
    """

    advice: StrategyAdvice | None = None
    expanded: bool = False
    action_text: str = ""
    equity_text: str = ""
    sizing_text: str = ""
    reasoning_text: str = ""
    draw_text: str = ""
    action_color: ActionColor = ActionColor.WHITE


class SolverPanel:
    """Display panel for GTO strategy recommendations.

    Shows the recommended action, equity, sizing, opponent ranges,
    and draw information in compact or expanded view.

    Args:
        expanded: Whether to start in expanded view.
    """

    def __init__(self, expanded: bool = False) -> None:
        self._state = SolverDisplayState(expanded=expanded)

    @property
    def state(self) -> SolverDisplayState:
        """Return the current display state."""
        return self._state

    @property
    def expanded(self) -> bool:
        """Return whether the panel is in expanded view."""
        return self._state.expanded

    @expanded.setter
    def expanded(self, value: bool) -> None:
        """Toggle expanded/compact view and reformat display."""
        self._state.expanded = value
        if self._state.advice is not None:
            self._format_advice(self._state.advice)

    def toggle_expanded(self) -> None:
        """Toggle between compact and expanded view."""
        self.expanded = not self._state.expanded

    def update_advice(self, advice: StrategyAdvice) -> None:
        """Update the panel with new strategy advice.

        Args:
            advice: New strategy advice from the solver.
        """
        self._state.advice = advice
        self._format_advice(advice)

        logger.debug(
            "Solver panel updated: %s (equity=%.1f%%)",
            self._state.action_text,
            advice.equity * 100,
        )

    def clear(self) -> None:
        """Clear the panel display."""
        self._state.advice = None
        self._state.action_text = ""
        self._state.equity_text = ""
        self._state.sizing_text = ""
        self._state.reasoning_text = ""
        self._state.draw_text = ""
        self._state.action_color = ActionColor.WHITE

    def get_compact_text(self) -> str:
        """Get the compact view text for overlay display.

        Returns:
            Single-line summary: "RAISE | Eq:65% Size:75%"
        """
        if self._state.advice is None:
            return ""

        action = self._state.action_text
        equity = self._state.equity_text

        parts = [action, equity]
        if self._state.sizing_text:
            parts.append(self._state.sizing_text)

        return " | ".join(parts)

    def get_expanded_text(self) -> str:
        """Get the expanded view text for overlay display.

        Returns:
            Multi-line detailed view with all available information.
        """
        if self._state.advice is None:
            return ""

        lines = [
            f"Action: {self._state.action_text}",
            f"Equity: {self._state.equity_text}",
        ]

        if self._state.sizing_text:
            lines.append(f"Sizing: {self._state.sizing_text}")

        if self._state.draw_text:
            lines.append(f"Draws: {self._state.draw_text}")

        advice = self._state.advice
        if advice.reasoning:
            lines.append(f"Note: {'; '.join(advice.reasoning)}")

        return "\n".join(lines)

    def get_display_text(self) -> str:
        """Get the current display text based on view mode.

        Returns:
            Formatted text for compact or expanded view.
        """
        if self._state.expanded:
            return self.get_expanded_text()
        return self.get_compact_text()

    def get_action_color(self) -> ActionColor:
        """Get the current action color.

        Returns:
            Color category for the current action recommendation.
        """
        return self._state.action_color

    def _format_advice(self, advice: StrategyAdvice) -> None:
        """Format advice into display strings.

        Args:
            advice: Strategy advice to format.
        """
        rec = advice.recommendation
        self._state.action_text = format_action(rec)
        self._state.action_color = get_action_color(rec)
        self._state.equity_text = f"Eq:{advice.equity * 100:.0f}%"

        if advice.recommended_sizing is not None:
            self._state.sizing_text = (
                f"Size:{advice.recommended_sizing * 100:.0f}%"
            )
        else:
            self._state.sizing_text = ""

        if advice.reasoning:
            self._state.reasoning_text = "; ".join(advice.reasoning)
        else:
            self._state.reasoning_text = ""

        self._state.draw_text = ""

    @staticmethod
    def _format_draw_info(draw_info: DrawInfo) -> str:
        """Format draw information for display.

        Args:
            draw_info: Draw information from the solver.

        Returns:
            Formatted draw string.
        """
        if draw_info.draw_description:
            return f"{draw_info.draw_description} ({draw_info.outs} outs)"
        if draw_info.outs > 0:
            return f"{draw_info.outs} outs ({draw_info.probability:.0%})"
        return ""
