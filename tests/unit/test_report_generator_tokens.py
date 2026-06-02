"""Token accumulation: build_report() should sum usage across all Claude calls
and surface input_tokens_total / output_tokens_total / cache_read_input_tokens_total
in report.metadata so the PostHog cost event has real numbers."""
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(
        profile=profile, df=df, claude=client,
        model_selection="m1", model_narrative="m2",
    )


def _three_chart_response(usage_selection=None, usage_narrative=None):
    """One selection call (no errors) + one narrative call, with custom usage.

    3 charts is under MIN_CHARTS_TARGET, so generate_charts() fires one reach-for-more
    round between selection and narrative. We script it as an empty round with ZERO
    usage so it adds no charts and does not perturb the token totals under assertion.
    """
    return FakeClaude([
        {
            "tool_calls": [
                tool_use("frequency_bar_chart",
                         {"column": "activity_type", "title": "T1", "intent": "i1"}),
                tool_use("histogram_chart",
                         {"column": "duration_minutes", "title": "T2", "intent": "i2"}),
                tool_use("aggregation_bar_chart",
                         {"value_col": "duration_minutes", "group_col": "activity_type",
                          "agg": "mean", "title": "T3", "intent": "i3"}),
            ],
            "usage": usage_selection or {},
        },
        {
            "tool_calls": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
        },
        {
            "tool_calls": [tool_use(
                "submit_narrative",
                {"summary": "s", "captions": ["c1", "c2", "c3"], "data_quality": []},
            )],
            "usage": usage_narrative or {},
        },
    ])


def test_build_report_metadata_includes_token_total_keys(activities):
    fake = _three_chart_response()
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert "input_tokens_total" in report.metadata
    assert "output_tokens_total" in report.metadata
    assert "cache_read_input_tokens_total" in report.metadata


def test_build_report_sums_tokens_across_selection_and_narrative(activities):
    fake = _three_chart_response(
        usage_selection={"input_tokens": 1500, "output_tokens": 400,
                         "cache_read_input_tokens": 200},
        usage_narrative={"input_tokens": 800, "output_tokens": 250,
                         "cache_read_input_tokens": 600},
    )
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert report.metadata["input_tokens_total"] == 1500 + 800
    assert report.metadata["output_tokens_total"] == 400 + 250
    assert report.metadata["cache_read_input_tokens_total"] == 200 + 600


def test_build_report_includes_retry_call_in_totals(activities):
    """When selection has a tool error, the retry call's tokens are also summed."""
    fake = FakeClaude([
        {
            "tool_calls": [
                tool_use("frequency_bar_chart",
                         {"column": "nope", "title": "Bad", "intent": "fail"},
                         id_="bad1"),
                tool_use("frequency_bar_chart",
                         {"column": "activity_type", "title": "Good", "intent": "good"},
                         id_="good1"),
            ],
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        },
        {
            "tool_calls": [tool_use("histogram_chart",
                                    {"column": "duration_minutes",
                                     "title": "Fixed", "intent": "fixed"})],
            "usage": {"input_tokens": 500, "output_tokens": 100},
        },
        {
            "tool_calls": [tool_use("submit_narrative",
                                    {"summary": "s", "captions": ["c"],
                                     "data_quality": []})],
            "usage": {"input_tokens": 300, "output_tokens": 80},
        },
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert report.metadata["input_tokens_total"] == 1000 + 500 + 300
    assert report.metadata["output_tokens_total"] == 200 + 100 + 80
