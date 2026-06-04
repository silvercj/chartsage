"""report_writer rolls up PASS/WARN/FAIL and writes report.md + summary.json."""
import json

from qa.report_writer import classify, write_report


def _payload(name, det_issues=None, judge=None, error=None, report=True):
    return {
        "name": name,
        "error": error,
        "elapsed_ms": 10,
        "rows_analyzed": 5, "cols_analyzed": 2,
        "was_sampled": False, "original_rows": 5,
        "report": {"summary": "s", "charts": []} if report else None,
        "deterministic_issues": det_issues or [],
        "judge": judge,
    }


def test_classify_fail_on_deterministic_fail():
    p = _payload("d", det_issues=[{"severity": "fail", "code": "x", "message": "m", "chart_id": None}])
    assert classify(p) == "FAIL"


def test_classify_fail_on_judge_nonsense():
    judge = {"charts": [{"chart_id": "c1", "makes_sense": False, "severity": "fail", "issue": "bad"}],
             "narrative_matches": True}
    assert classify(_payload("d", judge=judge)) == "FAIL"


def test_classify_warn_on_warn_only():
    p = _payload("d", det_issues=[{"severity": "warn", "code": "x", "message": "m", "chart_id": None}])
    assert classify(p) == "WARN"


def test_classify_warn_on_narrative_mismatch():
    judge = {"charts": [], "narrative_matches": False}
    assert classify(_payload("d", judge=judge)) == "WARN"


def test_classify_pass_when_clean():
    assert classify(_payload("d")) == "PASS"


def test_write_report_emits_files_and_table(tmp_path):
    payloads = [
        _payload("good"),
        _payload("bad", det_issues=[{"severity": "fail", "code": "degenerate_chart",
                                     "message": "boom", "chart_id": "c1"}]),
    ]
    write_report(tmp_path, payloads)
    md = (tmp_path / "report.md").read_text()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "| good | PASS |" in md
    assert "| bad | FAIL |" in md
    assert summary["totals"]["FAIL"] == 1
    assert summary["totals"]["PASS"] == 1
    assert summary["datasets"]["bad"]["status"] == "FAIL"
