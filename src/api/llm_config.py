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


# Per 1M tokens, in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cache_read": 0.1},
    "claude-sonnet-4-6":         {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "claude-opus-4-7":           {"input": 15.0, "output": 75.0, "cache_read": 1.5},
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost of a Claude API call from token counts.

    cache_read_tokens are billed at a fraction of the input rate.
    Returns USD to 6 decimal places.
    """
    rates = MODEL_PRICING.get(model, MODEL_PRICING["claude-haiku-4-5-20251001"])
    uncached_input = max(0, input_tokens - cache_read_tokens)
    cost = (
        uncached_input * rates["input"] / 1_000_000
        + cache_read_tokens * rates["cache_read"] / 1_000_000
        + output_tokens * rates["output"] / 1_000_000
    )
    return round(cost, 6)
