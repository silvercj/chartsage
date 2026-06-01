from schemas import KeyMetric, Report

def test_keymetric_defaults_number():
    m = KeyMetric(label="Revenue", value=1240.5)
    assert m.format == "number"

def test_report_key_metrics_defaults_empty():
    r = Report(generated_at="t", summary="s", data_quality=[], charts=[])
    assert r.key_metrics == []
