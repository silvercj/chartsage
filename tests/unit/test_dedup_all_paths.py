"""One chart identity (chart_signature: kind + source columns) deduped on EVERY path
that can produce specs — within a selection round, retry vs. initial, the reach-for-more
round, deepen vs. the seed, and generate_more vs. the existing report. Previously only
the fallback and the more-round deduped; the initial round, deepen and generate_more
relied purely on prompt instructions not to repeat."""
import pandas as pd
from unittest.mock import MagicMock

from report_generator import ReportGenerator
from profile import profile_dataframe
from schemas import ChartWithCaption
from tests.helpers.fake_claude import FakeClaude, tool_use


def _df():
    return pd.DataFrame({
        "activity_type": ["run", "run", "ride", "ride", "swim", "swim"] * 4,
        "duration": [30.0, 45.0, 60.0, 90.0, 25.0, 40.0] * 4,
    })


def _make_generator(df, fake):
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(
        profile=profile_dataframe(df), df=df, claude=client,
        model_selection="m1", model_narrative="m2",
    )


def _freq(title="t"):
    return tool_use("frequency_bar_chart",
                    {"column": "activity_type", "title": title, "intent": "i"})


def _hist(title="t"):
    return tool_use("histogram_chart",
                    {"column": "duration", "title": title, "intent": "i"})


def _box(title="t"):
    return tool_use("box_plot",
                    {"value_col": "duration", "group_col": "activity_type",
                     "title": title, "intent": "i"})


def test_same_signature_deduped_within_a_round():
    fake = FakeClaude([{"tool_calls": [_freq("first"), _freq("repeat"), _hist()]}])
    gen = _make_generator(_df(), fake)
    blocks = fake(model="m")  # consume scripted response to get the content blocks
    specs, errors = gen._execute_tool_calls(blocks.content)
    assert [s.title for s in specs] == ["first", "t"]   # the model's first wins
    assert errors == []


def test_retry_round_cannot_reintroduce_an_existing_chart():
    bad_call = tool_use("histogram_chart",
                        {"column": "missing_col", "title": "t", "intent": "i"})
    fake = FakeClaude([
        {"tool_calls": [_freq("first"), bad_call]},               # initial: 1 good + 1 error
        {"tool_calls": [_freq("dupe"), _box("new")]},             # retry repeats + adds new
        {"tool_calls": []},                                       # reach-for-more: nothing
    ])
    gen = _make_generator(_df(), fake)
    specs = gen.generate_charts()
    titles = [s.title for s in specs]
    assert "dupe" not in titles
    assert {"first", "new"} <= set(titles)


def test_deepen_stops_when_round_only_repeats():
    fake = FakeClaude([
        {"tool_calls": [_freq("repeat of seed")]},   # deepen round 1: only a repeat
    ])
    gen = _make_generator(_df(), fake)
    seed_fake = FakeClaude([{"tool_calls": [_freq("seed")]}])
    seed_blocks = seed_fake(model="m")
    seed, _ = gen._execute_tool_calls(seed_blocks.content)
    added = gen.deepen(seed)
    assert added == []
    assert len(fake.calls) == 1                      # a repeats-only round ends the loop


def test_generate_more_dedupes_against_existing_charts():
    fake = FakeClaude([
        {"tool_calls": [_freq("dupe"), _hist("fresh")]},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "s", "captions": ["c"], "data_quality": []})]},
    ])
    gen = _make_generator(_df(), fake)
    existing_fake = FakeClaude([{"tool_calls": [_freq("existing")]}])
    existing_specs, _ = gen._execute_tool_calls(existing_fake(model="m").content)
    existing = [ChartWithCaption(chart_id="c1", spec=existing_specs[0], caption="c")]
    new_charts, _layout = gen.generate_more(existing)
    assert [c.spec.title for c in new_charts] == ["fresh"]
