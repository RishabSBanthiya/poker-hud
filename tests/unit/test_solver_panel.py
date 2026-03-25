"""Unit tests for the SolverPanel overlay widget."""

from __future__ import annotations

from src.overlay.solver_panel import (
    ActionColor,
    SolverPanel,
    format_action,
    get_action_color,
)
from src.solver.advisor_coordinator import (
    ActionRecommendation,
    DrawInfo,
    StrategyAdvice,
)

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_advice(
    recommendation: ActionRecommendation = ActionRecommendation.CALL,
    equity: float = 0.55,
    recommended_sizing: float | None = None,
    reasoning: list[str] | None = None,
) -> StrategyAdvice:
    """Create a StrategyAdvice for testing."""
    return StrategyAdvice(
        recommendation=recommendation,
        equity=equity,
        recommended_sizing=recommended_sizing,
        reasoning=reasoning or [],
    )


# ---------------------------------------------------------------------------
# Action color/format tests
# ---------------------------------------------------------------------------


class TestActionHelpers:
    """Tests for action color and format helpers."""

    def test_raise_is_green(self) -> None:
        assert get_action_color(ActionRecommendation.RAISE) == ActionColor.GREEN

    def test_all_in_is_green(self) -> None:
        assert get_action_color(ActionRecommendation.ALL_IN) == ActionColor.GREEN

    def test_call_is_yellow(self) -> None:
        assert get_action_color(ActionRecommendation.CALL) == ActionColor.YELLOW

    def test_check_is_yellow(self) -> None:
        assert get_action_color(ActionRecommendation.CHECK) == ActionColor.YELLOW

    def test_fold_is_red(self) -> None:
        assert get_action_color(ActionRecommendation.FOLD) == ActionColor.RED

    def test_format_action_raise(self) -> None:
        assert format_action(ActionRecommendation.RAISE) == "RAISE"

    def test_format_action_fold(self) -> None:
        assert format_action(ActionRecommendation.FOLD) == "FOLD"

    def test_format_action_all_in(self) -> None:
        assert format_action(ActionRecommendation.ALL_IN) == "ALL_IN"


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
        advice = _make_advice(
            recommendation=ActionRecommendation.CALL, equity=0.55
        )
        self.panel.update_advice(advice)

        assert self.panel.state.action_text == "CALL"
        assert self.panel.state.equity_text == "Eq:55%"
        assert self.panel.get_action_color() == ActionColor.YELLOW

    def test_update_advice_raise(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.RAISE,
            equity=0.72,
            recommended_sizing=0.75,
        )
        self.panel.update_advice(advice)

        assert self.panel.state.action_text == "RAISE"
        assert self.panel.get_action_color() == ActionColor.GREEN

    def test_update_advice_fold(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.FOLD, equity=0.15
        )
        self.panel.update_advice(advice)

        assert self.panel.state.action_text == "FOLD"
        assert self.panel.get_action_color() == ActionColor.RED

    def test_compact_text(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.RAISE,
            equity=0.65,
            recommended_sizing=0.75,
        )
        self.panel.update_advice(advice)

        text = self.panel.get_compact_text()
        assert "RAISE" in text
        assert "Eq:65%" in text
        assert "Size:75%" in text

    def test_compact_text_no_sizing(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.FOLD, equity=0.20
        )
        self.panel.update_advice(advice)

        text = self.panel.get_compact_text()
        assert "FOLD" in text
        assert "Size" not in text

    def test_expanded_text(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.CALL,
            equity=0.55,
            reasoning=["Good pot odds"],
        )
        self.panel.update_advice(advice)

        text = self.panel.get_expanded_text()
        assert "Action: CALL" in text
        assert "Equity: Eq:55%" in text

    def test_expanded_text_with_reasoning(self) -> None:
        advice = _make_advice(
            reasoning=["Strong value bet opportunity"],
        )
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
            outs=9,
            draw_description="Nut flush draw",
        )
        result = SolverPanel._format_draw_info(draw)
        assert "Nut flush draw (9 outs)" in result

    def test_draw_format_with_outs_only(self) -> None:
        draw = DrawInfo(outs=9, probability=0.35)
        result = SolverPanel._format_draw_info(draw)
        assert "9 outs" in result
        assert "35%" in result

    def test_draw_format_no_draws(self) -> None:
        draw = DrawInfo(outs=0)
        result = SolverPanel._format_draw_info(draw)
        assert result == ""

    def test_expanded_mode_from_constructor(self) -> None:
        panel = SolverPanel(expanded=True)
        assert panel.expanded is True

    def test_sizing_text_present_when_sizing_given(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.RAISE,
            equity=0.70,
            recommended_sizing=0.50,
        )
        self.panel.update_advice(advice)
        assert self.panel.state.sizing_text == "Size:50%"

    def test_sizing_text_empty_when_no_sizing(self) -> None:
        advice = _make_advice(
            recommendation=ActionRecommendation.CHECK, equity=0.40
        )
        self.panel.update_advice(advice)
        assert self.panel.state.sizing_text == ""
