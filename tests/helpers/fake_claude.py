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
        resp.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return resp


def tool_use(name: str, input_dict: dict, id_: str = None) -> dict:
    """Convenience helper for building scripted tool calls."""
    out = {"name": name, "input": input_dict}
    if id_:
        out["id"] = id_
    return out
