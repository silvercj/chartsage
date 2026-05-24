"""Tests for the Claude client wrapper using mocks (no real API calls)."""
from unittest.mock import MagicMock, patch
import pytest
from claude_client import ClaudeClient, RetryableBusy


def make_response(content_blocks=None, usage=None):
    r = MagicMock()
    r.content = content_blocks or []
    r.usage = usage or MagicMock(input_tokens=100, output_tokens=50,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0)
    r.model = "claude-haiku-4-5-20251001"
    return r


def test_simple_call_returns_response():
    fake_sdk = MagicMock()
    fake_sdk.messages.create.return_value = make_response()

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk)
    resp = client.messages_create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp is not None
    fake_sdk.messages.create.assert_called_once()


def test_caches_system_and_tools_when_requested():
    fake_sdk = MagicMock()
    fake_sdk.messages.create.return_value = make_response()

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk)
    client.messages_create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system="You are a helper.",
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        messages=[{"role": "user", "content": "hi"}],
        cache_static=True,
    )

    call_args = fake_sdk.messages.create.call_args.kwargs
    assert call_args["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_args["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_retries_on_transient_5xx():
    from anthropic import APIStatusError
    err = APIStatusError("server error", response=MagicMock(status_code=502), body=None)
    fake_sdk = MagicMock()
    fake_sdk.messages.create.side_effect = [err, err, make_response()]

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk, _sleep=lambda s: None)
    resp = client.messages_create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp is not None
    assert fake_sdk.messages.create.call_count == 3


def test_surfaces_529_as_retryable_busy():
    from anthropic import APIStatusError
    err = APIStatusError("overloaded", response=MagicMock(status_code=529), body=None)
    fake_sdk = MagicMock()
    fake_sdk.messages.create.side_effect = err

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk, _sleep=lambda s: None)
    with pytest.raises(RetryableBusy):
        client.messages_create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )


def test_gives_up_after_max_retries():
    from anthropic import APIStatusError
    err = APIStatusError("server error", response=MagicMock(status_code=502), body=None)
    fake_sdk = MagicMock()
    fake_sdk.messages.create.side_effect = err

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk, _sleep=lambda s: None)
    with pytest.raises(APIStatusError):
        client.messages_create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert fake_sdk.messages.create.call_count == 3  # max attempts
