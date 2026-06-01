"""Anthropic tool definitions for chart selection and narrative."""


def _t(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_TITLE_INTENT = {
    "title": {"type": "string", "description": "Short chart title for the user."},
    "intent": {"type": "string", "description": "One-sentence rationale for this chart."},
}


CHART_TOOLS: list[dict] = [
    _t(
        "key_metrics",
        "Headline numbers shown as a stat band at the top of the report. Call this ONCE with the "
        "3–5 most important figures a reader wants first (a total, an average, a key rate, a notable count). "
        "You choose the label/column/agg; the value is computed from the data.",
        {
            "metrics": {
                "type": "array",
                "minItems": 1, "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short human label, e.g. 'Total revenue'."},
                        "column": {"type": "string"},
                        "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count", "nunique"]},
                        "format": {"type": "string", "enum": ["number", "currency", "percent"]},
                    },
                    "required": ["label", "column", "agg"],
                    "additionalProperties": False,
                },
            },
        },
        ["metrics"],
    ),
    _t(
        "frequency_bar_chart",
        "Bar chart of counts per category. Use for: 'how many rows fall in each category'.",
        {
            "column": {"type": "string", "description": "Categorical column to count by."},
            **_TITLE_INTENT,
        },
        ["column", "title", "intent"],
    ),
    _t(
        "aggregation_bar_chart",
        "Bar chart of an aggregation of a numeric column grouped by a categorical column. "
        "Use for: 'sum/mean/median/min/max of X by Y'. Does NOT support count — use frequency_bar_chart instead.",
        {
            "value_col": {"type": "string", "description": "Numeric column to aggregate."},
            "group_col": {"type": "string", "description": "Categorical column to group by."},
            "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max"]},
            **_TITLE_INTENT,
        },
        ["value_col", "group_col", "agg", "title", "intent"],
    ),
    _t(
        "histogram_chart",
        "Histogram of a numeric column's distribution. Use for: 'how is X distributed'.",
        {
            "column": {"type": "string", "description": "Numeric column to bin."},
            **_TITLE_INTENT,
        },
        ["column", "title", "intent"],
    ),
    _t(
        "scatter_chart",
        "Scatter plot of two numeric columns. Use for: 'is there a relationship between X and Y'. "
        "Optional color_by categorical column.",
        {
            "x_col": {"type": "string"},
            "y_col": {"type": "string"},
            "color_by": {"type": "string", "description": "Optional categorical column for coloring points."},
            **_TITLE_INTENT,
        },
        ["x_col", "y_col", "title", "intent"],
    ),
    _t(
        "line_chart",
        "Line chart over time. Use for trends. Aggregates a value column (or counts) by a time granularity.",
        {
            "date_col": {"type": "string"},
            "value_col": {"type": "string", "description": "Numeric column to aggregate (ignored when agg='count')."},
            "agg": {"type": "string", "enum": ["count", "sum", "mean", "median", "min", "max"]},
            "granularity": {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]},
            "group_by": {"type": "string", "description": "Optional categorical column for multiple lines."},
            **_TITLE_INTENT,
        },
        ["date_col", "value_col", "agg", "granularity", "title", "intent"],
    ),
    _t(
        "pie_chart",
        "Pie chart of composition. Best for ≤8 categories. Larger sets get rolled into 'Other'.",
        {
            "category_col": {"type": "string"},
            "value_col": {"type": "string", "description": "Numeric column to aggregate (omit when agg='count')."},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"]},
            **_TITLE_INTENT,
        },
        ["category_col", "agg", "title", "intent"],
    ),
    _t(
        "box_plot",
        "Box plot of a numeric column, optionally grouped by a categorical column.",
        {
            "value_col": {"type": "string"},
            "group_col": {"type": "string", "description": "Optional categorical column for groups."},
            **_TITLE_INTENT,
        },
        ["value_col", "title", "intent"],
    ),
    _t(
        "heatmap_chart",
        "Heatmap. Two modes: 'correlation' (correlation matrix of all numeric columns) "
        "or 'pivot' (aggregation of value_col by row_col × col_col).",
        {
            "mode": {"type": "string", "enum": ["correlation", "pivot"]},
            "row_col": {"type": "string", "description": "Required when mode='pivot'."},
            "col_col": {"type": "string", "description": "Required when mode='pivot'."},
            "value_col": {"type": "string", "description": "Required when mode='pivot' and agg != 'count'."},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"], "description": "Required when mode='pivot'."},
            **_TITLE_INTENT,
        },
        ["mode", "title", "intent"],
    ),
]


NARRATIVE_TOOL: dict = {
    "name": "submit_narrative",
    "description": "Submit the final report narrative.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 paragraph executive summary of what's in the data.",
            },
            "captions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One caption per chart, in the same order as the input charts.",
            },
            "data_quality": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Notes about data issues. Empty array if none.",
            },
        },
        "required": ["summary", "captions", "data_quality"],
        "additionalProperties": False,
    },
}
