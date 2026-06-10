"""Mock-able Claude response factory for integration tests.

Use:
    fake = FakeClaude(scripted_responses=[
        {"tool_calls": [{"name": "frequency_bar_chart", "input": {...}}]},
        {"tool_use": "submit_narrative", "input": {"summary": "...", "captions": [...], "data_quality": []}},
    ])
    client.messages_create = fake  # monkey-patch ClaudeClient
"""
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


@dataclass
class _ScriptedResponse:
    tool_calls: list[dict] = field(default_factory=list)
    text: str = ""


class FakeClaude:
    """Callable that returns canned responses in sequence."""

    def __init__(self, scripted: list[dict]):
        self.scripted = scripted
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        if idx >= len(self.scripted):
            raise AssertionError(f"FakeClaude received call #{idx+1} but only {len(self.scripted)} scripted")
        scripted = self.scripted[idx]

        content_blocks: list[Any] = []
        for tc in scripted.get("tool_calls", []):
            block = MagicMock()
            block.type = "tool_use"
            block.id = tc.get("id", f"tu_{idx}_{len(content_blocks)}")
            block.name = tc["name"]
            block.input = tc["input"]
            content_blocks.append(block)
        if scripted.get("text"):
            block = MagicMock()
            block.type = "text"
            block.text = scripted["text"]
            content_blocks.append(block)

        resp = MagicMock()
        resp.content = content_blocks
        resp.model = "claude-haiku-4-5-20251001"
        usage_overrides = scripted.get("usage", {})
        resp.usage = MagicMock(
            input_tokens=usage_overrides.get("input_tokens", 100),
            output_tokens=usage_overrides.get("output_tokens", 50),
            cache_read_input_tokens=usage_overrides.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage_overrides.get("cache_creation_input_tokens", 0),
        )
        return resp


def tool_use(name: str, input_dict: dict, id_: str = None) -> dict:
    """Convenience helper for building scripted tool calls."""
    out = {"name": name, "input": input_dict}
    if id_:
        out["id"] = id_
    return out


def ten_distinct_chart_calls(title_prefix: str = "T", intent_prefix: str = "i") -> list[dict]:
    """10 chart tool calls with 10 DISTINCT signatures (kind + source columns) over the
    `sales` fixture (region / revenue / order_date). The generator dedupes same-signature
    repeats and rejects degenerate charts, so scripted selections must vary their charts
    for all 10 to count. Titles are '<title_prefix>0'..'<title_prefix>9', ids tu_0..tu_9."""
    specs = [
        ("frequency_bar_chart", {"column": "region"}),
        ("aggregation_bar_chart", {"value_col": "revenue", "group_col": "region", "agg": "sum"}),
        ("histogram_chart", {"column": "revenue"}),
        ("box_plot", {"value_col": "revenue"}),
        ("pie_chart", {"category_col": "region", "agg": "count"}),
        ("pie_chart", {"category_col": "region", "value_col": "revenue", "agg": "sum"}),
        ("line_chart", {"date_col": "order_date", "value_col": "revenue",
                        "agg": "count", "granularity": "week"}),
        ("line_chart", {"date_col": "order_date", "value_col": "revenue",
                        "agg": "sum", "granularity": "week"}),
        ("treemap_chart", {"category_col": "region", "value_col": "revenue", "agg": "sum"}),
        ("dual_axis_chart", {"x_col": "region", "bar_value_col": "revenue",
                             "line_value_col": "revenue", "bar_agg": "sum", "line_agg": "mean"}),
    ]
    return [
        tool_use(name, {**inp, "title": f"{title_prefix}{i}", "intent": f"{intent_prefix}{i}"},
                 id_=f"tu_{i}")
        for i, (name, inp) in enumerate(specs)
    ]
