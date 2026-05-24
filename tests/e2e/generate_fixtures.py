"""Run once to (re)generate the e2e fixture CSVs."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)
rng = np.random.RandomState(42)


def activities():
    n = 200
    types = rng.choice(["consultation", "intro_call", "lab_test", "after_hours"], size=n, p=[0.4, 0.3, 0.25, 0.05])
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in rng.randint(0, 365, size=n)]
    durations = rng.normal(60, 25, size=n).clip(min=5)
    # Inject the anomalies the design spec mentions
    durations[0] = -30.0
    durations[1] = 600.0
    df = pd.DataFrame({
        "activity_id": range(1, n + 1),
        "patient_id": rng.randint(1001, 1100, size=n),
        "activity_type": types,
        "activity_date": dates,
        "duration_minutes": durations,
    })
    df.to_csv(OUT / "activities.csv", index=False)


def sales():
    n = 150
    regions = rng.choice(["north", "south", "east", "west"], size=n)
    rev = rng.lognormal(7, 1, size=n)
    dates = [datetime(2024, 1, 1) + timedelta(days=int(d)) for d in rng.randint(0, 180, size=n)]
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "region": regions,
        "revenue": rev,
        "order_date": dates,
        "product_category": rng.choice(["widgets", "gadgets", "doodads"], size=n),
    })
    df.to_csv(OUT / "sales.csv", index=False)


def signups():
    days = 90
    dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(days)]
    base = np.linspace(10, 50, days)
    seasonal = 10 * np.sin(np.linspace(0, 6, days))
    noise = rng.normal(0, 5, size=days)
    counts = (base + seasonal + noise).round().clip(min=0).astype(int)
    df = pd.DataFrame({"date": dates, "signups": counts})
    df.to_csv(OUT / "signups.csv", index=False)


def survey():
    n = 300
    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "satisfaction": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
        "would_recommend": rng.choice(["yes", "no", "maybe"], size=n, p=[0.55, 0.25, 0.2]),
        "channel": rng.choice(["web", "mobile", "store"], size=n),
    })
    df.to_csv(OUT / "survey.csv", index=False)


def degenerate():
    df = pd.DataFrame({"only_column": [1, None, None, 2, None] * 10})
    df.to_csv(OUT / "degenerate.csv", index=False)


if __name__ == "__main__":
    activities(); sales(); signups(); survey(); degenerate()
    print("fixtures written to", OUT)
