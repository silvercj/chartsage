import pandas as pd
from main import sample_for_analysis, MAX_ANALYSIS_ROWS


def test_large_frame_sampled_deterministically():
    df = pd.DataFrame({"a": range(120_000), "b": range(120_000)})
    out, was_sampled, total = sample_for_analysis(df)
    assert was_sampled is True and total == 120_000 and len(out) == MAX_ANALYSIS_ROWS
    out2, _, _ = sample_for_analysis(df)
    assert out.reset_index(drop=True).equals(out2.reset_index(drop=True))   # deterministic


def test_small_frame_untouched():
    df = pd.DataFrame({"a": range(100)})
    out, was_sampled, total = sample_for_analysis(df)
    assert was_sampled is False and total == 100 and len(out) == 100
