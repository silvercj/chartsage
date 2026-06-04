"""judge_report parses a forced-tool-call response into a JudgeVerdict (offline)."""
import types

from qa.judge import judge_report, JudgeVerdict
from tests.helpers.fake_claude import FakeClaude, tool_use


def _report():
    return {
        "summary": "Revenue rises over the year.",
        "charts": [
            {"chart_id": "c1", "caption": "trend", "spec": {
                "kind": "line", "title": "Revenue over time", "intent": "trend",
                "x": ["2019", "2020", "2021"], "y": [1.0, 2.0, 3.0],
                "x_label": "Year", "y_label": "Sum of revenue"}},
        ],
    }


def test_judge_parses_structured_verdict(monkeypatch):
    fake = FakeClaude([{"tool_calls": [tool_use("submit_judgement", {
        "charts": [{"chart_id": "c1", "makes_sense": True, "issue": None, "severity": "none"}],
        "narrative_matches": True,
    })]}])
    monkeypatch.setattr(
        "qa.judge._build_claude",
        lambda: types.SimpleNamespace(messages_create=fake),
    )
    verdict = judge_report("Rows: 3\nColumns ...", _report())
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.narrative_matches is True
    assert len(verdict.charts) == 1
    assert verdict.charts[0].chart_id == "c1"
    assert verdict.charts[0].makes_sense is True
    assert verdict.any_chart_fails is False


def test_judge_flags_nonsense_chart(monkeypatch):
    fake = FakeClaude([{"tool_calls": [tool_use("submit_judgement", {
        "charts": [{"chart_id": "c1", "makes_sense": False,
                    "issue": "line collapses to one point", "severity": "fail"}],
        "narrative_matches": False,
    })]}])
    monkeypatch.setattr(
        "qa.judge._build_claude",
        lambda: types.SimpleNamespace(messages_create=fake),
    )
    verdict = judge_report("profile", _report())
    assert verdict.any_chart_fails is True
    assert verdict.charts[0].severity == "fail"
