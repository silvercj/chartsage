import pandas as pd
import pytest
from chart_executor import execute_line_chart
from schemas import ChartSpec, ToolError


def _params(date_col="activity_date", value_col="duration_minutes",
            agg="count", granularity="month", group_by=None, title="t", intent="i"):
    p = {"date_col": date_col, "value_col": value_col, "agg": agg,
         "granularity": granularity, "title": title, "intent": intent}
    if group_by is not None:
        p["group_by"] = group_by
    return p


def test_count_by_month(activities):
    result = execute_line_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.kind == "line"
    assert len(result.x) == len(result.y)
    assert sum(result.y) == len(activities)


def test_mean_by_month(activities):
    result = execute_line_chart(activities, _params(agg="mean"))
    assert isinstance(result, ChartSpec)
    assert all(isinstance(v, float) for v in result.y)


def test_sum_by_quarter(activities):
    result = execute_line_chart(activities, _params(agg="sum", granularity="quarter"))
    assert isinstance(result, ChartSpec)


def test_with_group_by(activities):
    result = execute_line_chart(activities, _params(group_by="activity_type"))
    assert isinstance(result, ChartSpec)
    assert result.series is not None
    assert len(result.series) >= 1


def test_invalid_granularity(activities):
    result = execute_line_chart(activities, _params(granularity="century"))
    assert isinstance(result, ToolError)


def test_invalid_agg(activities):
    result = execute_line_chart(activities, _params(agg="nope"))
    assert isinstance(result, ToolError)


def test_non_date_column(activities):
    result = execute_line_chart(activities, _params(date_col="activity_type"))
    assert isinstance(result, ToolError)
    assert "date" in result.reason.lower()


def test_value_col_must_be_numeric_when_not_count(activities):
    result = execute_line_chart(activities, _params(value_col="activity_type", agg="mean"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_area_defaults_false(activities):
    result = execute_line_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.area is False


def test_area_flag_sets_spec_area(activities):
    params = _params(agg="count")
    params["area"] = True
    result = execute_line_chart(activities, params)
    assert isinstance(result, ChartSpec)
    assert result.area is True


def test_group_by_equal_to_value_col_is_ignored():
    # Grouping a line by the very column it measures is nonsensical: it yields one
    # one-point series per distinct value — a tangle of spikes with the x-axis mislabeled
    # with the values, not years. The executor must drop such a group_by and draw a single
    # clean series. Regression for the World Cup blowouts hero (group_by=blowout_pct on a
    # blowout_pct line) which rendered as a mess of coloured spikes.
    df = pd.DataFrame({
        "year": [2000, 2004, 2008, 2012, 2016, 2020],
        "blowout_pct": [10.0, 12.0, 8.0, 9.0, 7.0, 6.0],
    })
    result = execute_line_chart(df, _params(
        date_col="year", value_col="blowout_pct", agg="mean",
        granularity="year", group_by="blowout_pct"))
    assert isinstance(result, ChartSpec)
    assert result.series is None, "must collapse to a single series, not one-per-value"
    assert result.x and result.y and len(result.x) == 6


def test_integer_year_column_parsed_as_years():
    # Regression: a 4-digit integer year column must be treated as calendar years,
    # not nanoseconds-since-epoch (which collapsed every row to ~1970 -> a 1-point line).
    df = pd.DataFrame({
        "year": [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022],
        "goals": [2.2, 2.7, 2.7, 2.5, 2.3, 2.3, 2.7, 2.6, 2.7],
    })
    result = execute_line_chart(
        df, _params(date_col="year", value_col="goals", agg="mean", granularity="year"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) == 9
    assert result.x[0] == "1990" and result.x[-1] == "2022"
