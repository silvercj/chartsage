"""End-to-end smoke tests that hit real Claude.

Gated by RUN_E2E=true and a valid ANTHROPIC_API_KEY in the environment.
Run with: RUN_E2E=true pytest tests/e2e/ -m e2e -v
"""
import os
import pandas as pd
import pytest
from pathlib import Path
from claude_client import ClaudeClient
from llm_config import MODEL_SELECTION, MODEL_NARRATIVE
from profile import profile_dataframe
from report_generator import ReportGenerator
from schemas import Report


pytestmark = pytest.mark.e2e


FIXTURES = Path(__file__).parent / "fixtures"


def _skip_unless_enabled():
    if os.environ.get("RUN_E2E") != "true":
        pytest.skip("Set RUN_E2E=true to run e2e tests.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set.")


def _run(fixture_name: str) -> Report:
    _skip_unless_enabled()
    df = pd.read_csv(FIXTURES / fixture_name)
    df.columns = [c.lower() for c in df.columns]
    profile = profile_dataframe(df)
    client = ClaudeClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    gen = ReportGenerator(
        profile=profile, df=df, claude=client,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
    )
    return gen.build_report()


def test_activities_produces_report():
    report = _run("activities.csv")
    assert len(report.charts) >= 3
    assert report.summary
    for c in report.charts:
        assert c.spec.kind in {"bar", "histogram", "scatter", "line", "pie", "box", "heatmap"}
        if c.spec.x is not None and c.spec.y is not None:
            assert len(c.spec.x) == len(c.spec.y)


def test_activities_summary_mentions_anomaly():
    report = _run("activities.csv")
    text = (report.summary + " ".join(report.data_quality)).lower()
    assert "negative" in text or "outlier" in text or "extreme" in text


def test_sales_produces_report():
    report = _run("sales.csv")
    assert len(report.charts) >= 3


def test_signups_produces_line_chart_somewhere():
    report = _run("signups.csv")
    kinds = {c.spec.kind for c in report.charts}
    assert "line" in kinds or "bar" in kinds


def test_survey_produces_report():
    report = _run("survey.csv")
    assert len(report.charts) >= 3


def test_degenerate_does_not_crash():
    report = _run("degenerate.csv")
    assert report.summary or report.data_quality
