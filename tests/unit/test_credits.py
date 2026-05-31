import importlib


def test_default_costs():
    import credits
    importlib.reload(credits)
    assert credits.REPORT_COST == 100
    assert credits.GENERATE_MORE_COST == 40
    assert credits.SIGNUP_GRANT == 300


def test_costs_env_override(monkeypatch):
    monkeypatch.setenv("REPORT_COST", "250")
    monkeypatch.setenv("GENERATE_MORE_COST", "80")
    monkeypatch.setenv("SIGNUP_GRANT", "1000")
    import credits
    importlib.reload(credits)
    assert (credits.REPORT_COST, credits.GENERATE_MORE_COST, credits.SIGNUP_GRANT) == (250, 80, 1000)
    monkeypatch.undo()
    importlib.reload(credits)


def test_insufficient_credits_is_exception():
    import credits
    assert issubclass(credits.InsufficientCredits, Exception)
