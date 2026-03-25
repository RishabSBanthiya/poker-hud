"""Strategy advisor display panel for the overlay.

Displays GTO solver recommendations including recommended action,
equity, pot odds, opponent range estimation, and draw information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from src.solver.advisor_coordinator import DrawInfo, StrategyAdvice
from src.solver.postflop_advisor import Action

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
_ACTION_COLORS: dict[Action, ActionColor] = {
    Action.RAISE: ActionColor.GREEN,
    Action.ALL_IN: ActionColor.GREEN,
    Action.CALL: ActionColor.YELLOW,
    Action.CHECK: ActionColor.YELLOW,
    Action.FOLD: ActionColor.RED,
}


def get_action_color(action: Action) -> ActionColor:
    """Get the display color for an action.

    Args:
        action: The poker action.

    Returns:
        Color category for the action.
    """
    return _ACTION_COLORS.get(action, ActionColor.WHITE)


def format_action(action: Action) -> str:
    """Format an action for display.

    Args:
        action: The poker action.

    Returns:
        Human-readable action string.
    """
    return action.value.upper()


@dataclass
class SolverDisplayState:
    """Current display state of the solver panel.

    Attributes:
        advice: The current strategy advice, if any.
        expanded: Whether the panel is in expanded view.
        action_text: Formatted action recommendation text.
        equity_text: Formatted equity display text.
        pot_odds_text: Formatted pot odds display text.
        range_text: Opponent range summary text.
        draw_text: Draw information text.
        action_color: Color for the action display.
    """

    advice: StrategyAdvice | None = None
    expanded: bool = False
    action_text: str = ""
    equity_text: str = ""
    pot_odds_text: str = ""
    range_text: str = ""
    draw_text: str = ""
    action_color: ActionColor = ActionColor.WHITE


class SolverPanel:
    """Display panel for GTO strategy recommendations.

    Shows the recommended action, equity, pot odds, opponent range,
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
            "Solver panel updated: %s (equity=%.1f%%, pot_odds=%.1f%%)",
            self._state.action_text,
            advice.equity,
            advice.pot_odds,
        )

    def clear(self) -> None:
        """Clear the panel display."""
        self._state.advice = None
        self._state.action_text = ""
        self._state.equity_text = ""
        self._state.pot_odds_text = ""
        self._state.range_text = ""
        self._state.draw_text = ""
        self._state.action_color = ActionColor.WHITE

    def get_compact_text(self) -> str:
        """Get the compact view text for overlay display.

        Returns:
            Single-line summary: "RAISE | Eq:65% PO:33%"
        """
        if self._state.advice is None:
            return ""

        action = self._state.action_text
        equity = self._state.equity_text
        pot_odds = self._state.pot_odds_text

        return f"{action} | {equity} {pot_odds}"

    def get_expanded_text(self) -> str:
        """Get the expanded view text for overlay display.

        Returns:
            Multi-line detailed view with all available information.
        """
        if self._state.advice is None:
            return ""

        lines = [
            f"Action: {self._state.action_text}",
            f"Equity: {self._state.equity_text}  "
            f"Pot Odds: {self._state.pot_odds_text}",
        ]

        if self._state.range_text:
            lines.append(f"Range: {self._state.range_text}")

        if self._state.draw_text:
            lines.append(f"Draws: {self._state.draw_text}")

        advice = self._state.advice
        if advice.recommendation.reasoning:
            lines.append(f"Note: {advice.recommendation.reasoning}")

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
        self._state.action_text = format_action(rec.action)
        self._state.action_color = get_action_color(rec.action)
        self._state.equity_text = f"Eq:{advice.equity:.0f}%"
        self._state.pot_odds_text = f"PO:{advice.pot_odds:.0f}%"
        self._state.range_text = advice.opponent_range_summary

        if advice.draw_info is not None:
            self._state.draw_text = self._format_draw_info(advice.draw_info)
        else:
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

        parts = []
        if draw_info.has_flush_draw:
            parts.append("Flush draw")
        if draw_info.has_straight_draw:
            parts.append("Straight draw")
        if parts:
            return f"{', '.join(parts)} ({draw_info.outs} outs)"
        return ""
