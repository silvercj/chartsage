"""The request-a-chart action gets the same one-retry-round courtesy as the main
selection pass: a first attempt that errors (bad column, degenerate chart) is fed back
as tool_results so the model can correct itself, instead of immediately 422ing."""
import pandas as pd
from unittest.mock import MagicMock

from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def _df():
    return pd.DataFrame({
        "activity_type": ["run", "run", "ride", "ride", "swim", "swim"] * 4,
        "duration": [30.0, 45.0, 60.0, 90.0, 25.0, 40.0] * 4,
    })


def _make_generator(fake):
    df = _df()
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(
        profile=profile_dataframe(df), df=df, claude=client,
        model_selection="m1", model_narrative="m2",
    )


def test_add_chart_retries_once_after_an_error():
    fake = FakeClaude([
        {"tool_calls": [tool_use("histogram_chart",
                                 {"column": "missing", "title": "t", "intent": "i"})]},
        {"tool_calls": [tool_use("histogram_chart",
                                 {"column": "duration", "title": "fixed", "intent": "i"})]},
    ])
    gen = _make_generator(fake)
    cwc = gen.add_chart(mode="describe", chart_type=None, prompt="distribution of duration")
    assert cwc is not None
    assert cwc.spec.title == "fixed"
    # The retry conversation must include the original add-chart instruction and the error.
    retry_messages = fake.calls[1]["messages"]
    assert "distribution of duration" in retry_messages[0]["content"]
    assert any(tr.get("is_error") for tr in retry_messages[2]["content"])


def test_add_chart_returns_none_when_retry_also_fails():
    fake = FakeClaude([
        {"tool_calls": [tool_use("histogram_chart",
                                 {"column": "missing", "title": "t", "intent": "i"})]},
        {"tool_calls": [tool_use("histogram_chart",
                                 {"column": "still_missing", "title": "t", "intent": "i"})]},
    ])
    gen = _make_generator(fake)
    assert gen.add_chart(mode="describe", chart_type=None, prompt="x") is None
