import os
import importlib
import pytest


def reload_config():
    import llm_config
    importlib.reload(llm_config)
    return llm_config


def test_default_is_haiku(monkeypatch):
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL_SELECTION", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL_NARRATIVE", raising=False)
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-haiku-4-5-20251001"
    assert cfg.MODEL_NARRATIVE == "claude-haiku-4-5-20251001"


def test_generic_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "sonnet-4-6")
    monkeypatch.delenv("CLAUDE_MODEL_SELECTION", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL_NARRATIVE", raising=False)
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-sonnet-4-6"
    assert cfg.MODEL_NARRATIVE == "claude-sonnet-4-6"


def test_per_pass_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "haiku-4-5")
    monkeypatch.setenv("CLAUDE_MODEL_NARRATIVE", "sonnet-4-6")
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-haiku-4-5-20251001"
    assert cfg.MODEL_NARRATIVE == "claude-sonnet-4-6"


def test_passthrough_full_id(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-7")
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-opus-4-7"


def test_resolve_unknown_alias_passes_through():
    cfg = reload_config()
    assert cfg.resolve("custom-id-string") == "custom-id-string"
