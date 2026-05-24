"""Small DataFrame factories for tests."""
from datetime import datetime, timedelta
import pandas as pd


def activities_df() -> pd.DataFrame:
    """The canonical example: categorical type + numeric duration + date + identifier."""
    return pd.DataFrame({
        "activity_id": list(range(1, 21)),
        "patient_id": [1001, 1002, 1003, 1001, 1002, 1003, 1004, 1005,
                       1001, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
                       1013, 1014, 1015, 1016],
        "activity_type": ["consultation", "consultation", "lab_test", "intro_call",
                          "consultation", "lab_test", "lab_test", "intro_call",
                          "consultation", "lab_test", "intro_call", "consultation",
                          "consultation", "lab_test", "intro_call", "consultation",
                          "lab_test", "intro_call", "consultation", "lab_test"],
        "activity_date": [datetime(2024, 1, 1) + timedelta(days=i * 3) for i in range(20)],
        "duration_minutes": [30.0, 45.0, 15.0, 20.0, 60.0, 25.0, 30.0, 15.0,
                             40.0, 20.0, 25.0, 50.0, 35.0, 30.0, 18.0, 55.0,
                             22.0, 19.0, 42.0, 28.0],
    })


def sales_df() -> pd.DataFrame:
    """Revenue / region / date — classic BI shape."""
    return pd.DataFrame({
        "order_id": list(range(1, 16)),
        "region": ["north", "south", "east", "west", "north",
                   "south", "east", "west", "north", "south",
                   "east", "west", "north", "south", "east"],
        "revenue": [1200.0, 850.0, 2100.0, 1500.0, 1800.0,
                    900.0, 2400.0, 1750.0, 1300.0, 950.0,
                    2200.0, 1650.0, 1400.0, 1000.0, 2300.0],
        "order_date": pd.date_range("2024-01-01", periods=15, freq="3D"),
    })


def degenerate_df() -> pd.DataFrame:
    """Single column, mostly null — stress test."""
    return pd.DataFrame({"x": [1.0, None, None, 2.0, None, None, None, 3.0]})


def negative_duration_df() -> pd.DataFrame:
    """Triggers the 'negative values in duration column' anomaly."""
    return pd.DataFrame({
        "activity_type": ["a", "b", "c", "a", "b"],
        "duration_minutes": [-30.0, 45.0, 60.0, 20.0, 30.0],
    })
