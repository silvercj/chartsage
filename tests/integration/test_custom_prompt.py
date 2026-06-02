"""Custom prompt threads into BOTH the selection and narrative user messages,
and persists onto the report metadata (so generate-more / deep analysis honor it).

It must reach the USER message (never the cached system prompt) so prompt
caching stays intact, and it's treated as guidance (it does not override rules).
"""
import types

from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


FOCUS = "focus on margins by region"


def _gen(df, fc):
    # FakeClaude is the callable bound to claude.messages_create; it records every
    # call's kwargs (incl. `messages`) into fc.calls.
    return ReportGenerator(
        profile=profile_dataframe(df), df=df,
        claude=types.SimpleNamespace(messages_create=fc),
        model_selection="m", model_narrative="m",
        custom_prompt=FOCUS,
    )


def test_custom_prompt_reaches_selection_and_narrative_user_messages(sales):
    fc = FakeClaude([
        {"tool_calls": [tool_use("frequency_bar_chart",
                                 {"column": "region", "title": "T", "intent": "i"})]},
        {"tool_calls": []},  # reach-for-more (1 chart < target, no errors) proposes nothing
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": ["c"], "data_quality": []})]},
    ])
    gen = _gen(sales, fc)
    gen.build_report()

    # Call #0 selection, call #1 reach-for-more, call #2 narrative.
    selection_msgs = " ".join(str(m) for m in fc.calls[0]["messages"])
    narrative_msgs = " ".join(str(m) for m in fc.calls[-1]["messages"])
    assert FOCUS in selection_msgs, "focus must reach the selection user message"
    assert FOCUS in narrative_msgs, "focus must reach the narrative user message"

    # Never injected into the cached system prompt.
    for call in fc.calls:
        assert FOCUS not in str(call.get("system", "")), "focus must NOT touch the system prompt"


def test_custom_prompt_persisted_to_metadata(sales):
    fc = FakeClaude([
        {"tool_calls": [tool_use("frequency_bar_chart",
                                 {"column": "region", "title": "T", "intent": "i"})]},
        {"tool_calls": []},  # reach-for-more (1 chart < target, no errors) proposes nothing
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": ["c"], "data_quality": []})]},
    ])
    report = _gen(sales, fc).build_report()
    assert report.metadata.get("custom_prompt") == FOCUS


def test_custom_prompt_in_generate_more_user_message(sales):
    fc = FakeClaude([
        {"tool_calls": [tool_use("frequency_bar_chart",
                                 {"column": "region", "title": "T", "intent": "i"})]},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": ["c"], "data_quality": []})]},
    ])
    gen = _gen(sales, fc)
    gen.generate_more([])
    more_msgs = " ".join(str(m) for m in fc.calls[0]["messages"])
    assert FOCUS in more_msgs


def test_blank_custom_prompt_is_none_and_not_injected(sales):
    fc = FakeClaude([
        {"tool_calls": [tool_use("frequency_bar_chart",
                                 {"column": "region", "title": "T", "intent": "i"})]},
        {"tool_calls": []},  # reach-for-more (1 chart < target, no errors) proposes nothing
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": ["c"], "data_quality": []})]},
    ])
    gen = ReportGenerator(
        profile=profile_dataframe(sales), df=sales,
        claude=types.SimpleNamespace(messages_create=fc),
        model_selection="m", model_narrative="m",
        custom_prompt="   ",
    )
    assert gen.custom_prompt is None
    report = gen.build_report()
    assert report.metadata.get("custom_prompt") is None
    # No stray focus phrasing leaks into the user messages when there's no prompt.
    for call in fc.calls:
        assert "User's focus" not in " ".join(str(m) for m in call["messages"])
