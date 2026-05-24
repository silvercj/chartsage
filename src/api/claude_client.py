"""Thin wrapper around the anthropic SDK.

Responsibilities:
- Exponential backoff retry on transient 5xx (max 3 attempts).
- Surface 529 (overloaded) as RetryableBusy so the API layer can return 503.
- Optionally cache_control system + tools (saves ~90% on input tokens for repeats).
- Log token usage per call (caller decides what to do with it).
"""
import logging
import time
from typing import Any, Callable, Optional

import anthropic
from anthropic import APIStatusError


class RetryableBusy(Exception):
    """Raised when Claude returns 529 (overloaded). The API layer maps this to HTTP 503."""


class ClaudeClient:
    MAX_ATTEMPTS = 3
    BACKOFF_SECONDS = (1.0, 2.0, 4.0)

    def __init__(self, api_key: str, _sdk: Any = None, _sleep: Callable[[float], None] = time.sleep):
        self._sdk = _sdk if _sdk is not None else anthropic.Anthropic(api_key=api_key)
        self._sleep = _sleep

    def messages_create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict] = None,
        cache_static: bool = False,
    ):
        """Call anthropic.messages.create with retries and optional caching."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            if cache_static:
                kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                kwargs["system"] = system

        if tools:
            if cache_static:
                tools_with_cache = [dict(t) for t in tools]
                tools_with_cache[-1] = {**tools_with_cache[-1], "cache_control": {"type": "ephemeral"}}
                kwargs["tools"] = tools_with_cache
            else:
                kwargs["tools"] = tools

        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        return self._call_with_retries(kwargs)

    def _call_with_retries(self, kwargs: dict):
        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self._sdk.messages.create(**kwargs)
                if hasattr(response, "usage"):
                    u = response.usage
                    logging.info(
                        "[CLAUDE] model=%s input=%d output=%d cache_read=%d cache_write=%d",
                        getattr(response, "model", "?"),
                        getattr(u, "input_tokens", 0),
                        getattr(u, "output_tokens", 0),
                        getattr(u, "cache_read_input_tokens", 0),
                        getattr(u, "cache_creation_input_tokens", 0),
                    )
                return response
            except APIStatusError as e:
                status = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
                if status == 529:
                    raise RetryableBusy("Claude API is overloaded") from e
                if 500 <= status < 600:
                    last_exc = e
                    if attempt < self.MAX_ATTEMPTS - 1:
                        self._sleep(self.BACKOFF_SECONDS[attempt])
                        continue
                raise
        assert last_exc is not None
        raise last_exc
