import pytest
from llm_config import MODEL_PRICING, estimate_cost_usd


def test_haiku_pricing_known():
    cost = estimate_cost_usd("claude-haiku-4-5-20251001",
                             input_tokens=1_000_000,
                             output_tokens=0,
                             cache_read_tokens=0)
    assert cost == pytest.approx(1.0)


def test_haiku_output_pricing():
    cost = estimate_cost_usd("claude-haiku-4-5-20251001",
                             input_tokens=0,
                             output_tokens=1_000_000,
                             cache_read_tokens=0)
    assert cost == pytest.approx(5.0)


def test_cache_read_cheaper_than_normal_input():
    no_cache = estimate_cost_usd("claude-haiku-4-5-20251001",
                                 input_tokens=1_000_000, output_tokens=0, cache_read_tokens=0)
    all_cached = estimate_cost_usd("claude-haiku-4-5-20251001",
                                   input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000)
    assert all_cached < no_cache
    assert all_cached == pytest.approx(0.1)


def test_unknown_model_falls_back_to_haiku():
    cost = estimate_cost_usd("some-mystery-model",
                             input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(1.0)


def test_combines_input_output_cache_read():
    cost = estimate_cost_usd("claude-haiku-4-5-20251001",
                             input_tokens=2_000_000,
                             output_tokens=500_000,
                             cache_read_tokens=500_000)
    # uncached input: 1.5M @ $1.0/M = $1.50
    # cache_read:     0.5M @ $0.10/M = $0.05
    # output:         0.5M @ $5.0/M = $2.50
    # total = $4.05
    assert cost == pytest.approx(4.05)


def test_model_pricing_has_three_models():
    assert "claude-haiku-4-5-20251001" in MODEL_PRICING
    assert "claude-sonnet-4-6" in MODEL_PRICING
    assert "claude-opus-4-7" in MODEL_PRICING
