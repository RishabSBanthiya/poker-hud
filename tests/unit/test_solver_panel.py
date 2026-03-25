"""Unit tests for the SolverPanel overlay widget."""

from __future__ import annotations

from src.overlay.solver_panel import (
    ActionColor,
    SolverPanel,
    format_action,
    get_action_color,
)
from src.solver.advisor_coordinator import DrawInfo, StrategyAdvice
from src.solver.postflop_advisor import Action, ActionRecommendation

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_advice(
    action: Action = Action.CALL,
    confidence: float = 0.8,
    ev: float = 1.5,
    equity: float = 55.0,
    pot_odds: float = 33.0,
    opponent_range: str = "Top 20%",
    draw_info: DrawInfo | None = None,
    reasoning: str = "",
) -> StrategyAdvice:
    """Create a StrategyAdvice for testing."""
    return StrategyAdvice(
        recommendation=ActionRecommendation(
            action=action,
            confidence=confidence,
            ev=ev,
            reasoning=reasoning,
        ),
        equity=equity,
        pot_odds=pot_odds,
        opponent_range_summary=opponent_range,
        draw_info=draw_info,
    )


# ---------------------------------------------------------------------------
# Action color/format tests
# ---------------------------------------------------------------------------


class TestActionHelpers:
    """Tests for action color and format helpers."""

    def test_raise_is_green(self) -> None:
        assert get_action_color(Action.RAISE) == ActionColor.GREEN

    def test_all_in_is_green(self) -> None:
        assert get_action_color(Action.ALL_IN) == ActionColor.GREEN

    def test_call_is_yellow(self) -> None:
        assert get_action_color(Action.CALL) == ActionColor.YELLOW

    def test_check_is_yellow(self) -> None:
        assert get_action_color(Action.CHECK) == ActionColor.YELLOW

    def test_fold_is_red(self) -> None:
        assert get_action_color(Action.FOLD) == ActionColor.RED

    def test_format_action_raise(self) -> None:
        assert format_action(Action.RAISE) == "RAISE"

    def test_format_action_fold(self) -> None:
        assert format_action(Action.FOLD) == "FOLD"

    def test_format_action_all_in(self) -> None:
        assert format_action(Action.ALL_IN) == "ALL_IN"


# ---------------------------------------------------------------------------
# SolverPanel tests
# ---------------------------------------------------------------------------


class TestSolverPanel:
    """Tests for SolverPanel display logic."""

    def setup_method(self) -> None:
        self.panel = SolverPanel()

    def test_initial_state(self) -> None:
        assert self.panel.state.advice is None
        assert self.panel.expanded is False
        assert self.panel.get_display_text() == ""
        assert self.panel.get_action_color() == ActionColor.WHITE

    def test_update_advice_call(self) -> None:
        advice = _make_advice(action=Action.CALL, equity=55.0, pot_odds=33.0)
        self.panel.update_advice(advice)

        assert self.panel.state.action_text == "CALL"
        assert self.panel.state.equity_text == "Eq:55%"
        assert self.panel.state.pot_odds_text == "PO:33%"
        assert self.panel.get_action_color() == ActionColor.YELLOW

    def test_update_advice_raise(self) -> None:
        advice = _make_advice(action=Action.RAISE, equity=72.0, pot_odds=25.0)
        self.panel.update_advice(advice)

        assert self.panel.state.action_text == "RAISE"
        assert self.panel.get_action_color() == ActionColor.GREEN

    def test_update_advice_fold(self) -> None:
        advice = _make_advice(action=Action.FOLD, equity=15.0, pot_odds=40.0)
        self.panel.update_advice(advice)

        assert self.panel.state.action_text == "FOLD"
        assert self.panel.get_action_color() == ActionColor.RED

    def test_compact_text(self) -> None:
        advice = _make_advice(action=Action.RAISE, equity=65.0, pot_odds=33.0)
        self.panel.update_advice(advice)

        text = self.panel.get_compact_text()
        assert "RAISE" in text
        assert "Eq:65%" in text
        assert "PO:33%" in text

    def test_expanded_text(self) -> None:
        advice = _make_advice(
            action=Action.CALL,
            equity=55.0,
            pot_odds=33.0,
            opponent_range="Top 20%",
        )
        self.panel.update_advice(advice)

        text = self.panel.get_expanded_text()
        assert "Action: CALL" in text
        assert "Equity: Eq:55%" in text
        assert "Range: Top 20%" in text

    def test_expanded_text_with_draws(self) -> None:
        draw = DrawInfo(
            has_flush_draw=True,
            has_straight_draw=False,
            outs=9,
            draw_description="Nut flush draw",
        )
        advice = _make_advice(draw_info=draw)
        self.panel.update_advice(advice)

        text = self.panel.get_expanded_text()
        assert "Draws:" in text
        assert "Nut flush draw" in text
        assert "9 outs" in text

    def test_expanded_text_with_reasoning(self) -> None:
        advice = _make_advice(reasoning="Strong value bet opportunity")
        self.panel.update_advice(advice)

        text = self.panel.get_expanded_text()
        assert "Note: Strong value bet opportunity" in text

    def test_toggle_expanded(self) -> None:
        advice = _make_advice()
        self.panel.update_advice(advice)

        assert self.panel.expanded is False
        self.panel.toggle_expanded()
        assert self.panel.expanded is True

        text = self.panel.get_display_text()
        assert "Action:" in text  # expanded format

    def test_get_display_text_uses_mode(self) -> None:
        advice = _make_advice()
        self.panel.update_advice(advice)

        # Compact
        compact = self.panel.get_display_text()
        assert "|" in compact

        # Expanded
        self.panel.expanded = True
        expanded = self.panel.get_display_text()
        assert "Action:" in expanded

    def test_clear(self) -> None:
        advice = _make_advice()
        self.panel.update_advice(advice)
        self.panel.clear()

        assert self.panel.state.advice is None
        assert self.panel.state.action_text == ""
        assert self.panel.get_display_text() == ""
        assert self.panel.get_action_color() == ActionColor.WHITE

    def test_draw_format_with_description(self) -> None:
        draw = DrawInfo(
            has_flush_draw=True,
            outs=9,
            draw_description="Nut flush draw",
        )
        advice = _make_advice(draw_info=draw)
        self.panel.update_advice(advice)
        assert "Nut flush draw (9 outs)" in self.panel.state.draw_text

    def test_draw_format_without_description(self) -> None:
        draw = DrawInfo(
            has_flush_draw=True,
            has_straight_draw=True,
            outs=15,
        )
        advice = _make_advice(draw_info=draw)
        self.panel.update_advice(advice)
        assert "Flush draw" in self.panel.state.draw_text
        assert "Straight draw" in self.panel.state.draw_text
        assert "15 outs" in self.panel.state.draw_text

    def test_draw_format_no_draws(self) -> None:
        draw = DrawInfo(has_flush_draw=False, has_straight_draw=False, outs=0)
        advice = _make_advice(draw_info=draw)
        self.panel.update_advice(advice)
        assert self.panel.state.draw_text == ""

    def test_no_draw_info(self) -> None:
        advice = _make_advice(draw_info=None)
        self.panel.update_advice(advice)
        assert self.panel.state.draw_text == ""

    def test_expanded_mode_from_constructor(self) -> None:
        panel = SolverPanel(expanded=True)
        assert panel.expanded is True
