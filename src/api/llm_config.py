"""Model alias resolution.

Env var resolution order (first non-empty wins):
1. Per-pass override: CLAUDE_MODEL_SELECTION / CLAUDE_MODEL_NARRATIVE
2. Generic: CLAUDE_MODEL
3. Default: 'haiku-4-5'
"""
import os


MODEL_ALIASES = {
    "haiku-4-5":  "claude-haiku-4-5-20251001",
    "sonnet-4-6": "claude-sonnet-4-6",
    "opus-4-7":   "claude-opus-4-7",
}


def resolve(alias_or_id: str) -> str:
    """Map a friendly alias to a full model ID; pass through unknown strings."""
    return MODEL_ALIASES.get(alias_or_id, alias_or_id)


def _pick(*candidates: str | None, default: str) -> str:
    for c in candidates:
        if c:
            return c
    return default


MODEL_SELECTION = resolve(_pick(
    os.getenv("CLAUDE_MODEL_SELECTION"),
    os.getenv("CLAUDE_MODEL"),
    default="haiku-4-5",
))

MODEL_NARRATIVE = resolve(_pick(
    os.getenv("CLAUDE_MODEL_NARRATIVE"),
    os.getenv("CLAUDE_MODEL"),
    default="haiku-4-5",
))
