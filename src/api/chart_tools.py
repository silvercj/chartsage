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

# Optional, never required: how the charted value displays. The model's declaration wins
# over column-name keyword sniffing — it knows from context that a football goal 'margin'
# is a plain number (not a %) and that 'gross_margin_pct' is a % (not $).
_Y_FORMAT = {
    "y_format": {
        "type": "string",
        "enum": ["number", "currency", "percent"],
        "description": "How the charted value should display. Set it when you know the unit "
                       "from context: 'percent' for a rate/share/proportion, 'currency' for "
                       "money, 'number' for a plain measure or count-like value. Omit to infer "
                       "from the column name.",
    },
}


CHART_TOOLS: list[dict] = [
    _t(
        "key_metrics",
        "Headline numbers shown as a stat band at the top of the report. Call this ONCE with the "
        "3–5 most important figures a reader wants first (a total, an average, a key rate, a notable count). "
        "Pick genuine MEASURES — never a year, date, row index, or ID as a headline number (e.g. NOT 'Current year: 2025'); "
        "label each accurately for its aggregation: a 'growth'/'change'/'increase' label must be a real delta (use a filter), not a sum or total. "
        "You choose the label/column/agg; the value is computed from the data over ALL rows by default. "
        "To describe ONE group instead (e.g. the win rate for host nations only, or sales in one region), "
        "add the optional `filter` {column, value} — the value is then computed only over rows where that "
        "column equals that value. IMPORTANT: without a filter the metric covers every row, so never write a "
        "label implying a subset (e.g. 'Host-nation win rate') unless you set the matching filter — otherwise "
        "the number contradicts its label. To compare two groups, emit two metrics with different filters.",
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
                        "filter": {
                            "type": "object",
                            "description": "Optional. Compute the metric over only the rows where `column` == `value` "
                                           "(e.g. {\"column\": \"venue\", \"value\": \"Host nation\"}). Omit for an all-rows metric.",
                            "properties": {
                                "column": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["column", "value"],
                            "additionalProperties": False,
                        },
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
        "Bar chart of counts per category. Use for: 'how many rows fall in each category'. "
        "With many categories it shows the top ones by count.",
        {
            "column": {"type": "string", "description": "Categorical column to count by."},
            **_TITLE_INTENT,
        },
        ["column", "title", "intent"],
    ),
    _t(
        "aggregation_bar_chart",
        "Bar chart of a numeric column aggregated by a categorical column — and the tool to RANK "
        "categories/entities by a value (e.g. 'mean score by team', 'total sales by product', "
        "'win rate by circuit'). Works even when there is one row per entity (use agg='mean' or 'sum'). "
        "With many categories it shows the top ones by value. Use for 'sum/mean/median/min/max of X by Y' "
        "and 'which Y has the highest/lowest X'. Does NOT support count — use frequency_bar_chart instead.",
        {
            "value_col": {"type": "string", "description": "Numeric column to aggregate."},
            "group_col": {"type": "string", "description": "Categorical column to group by."},
            "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max"]},
            **_Y_FORMAT,
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
            **_Y_FORMAT,
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
            "area": {"type": "boolean", "description": "Fill under the line (area chart). Multi-series + area = stacked area."},
            **_Y_FORMAT,
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
            **_Y_FORMAT,
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
            **_Y_FORMAT,
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
            **_Y_FORMAT,
            **_TITLE_INTENT,
        },
        ["mode", "title", "intent"],
    ),
    _t(
        "grouped_bar_chart",
        "Bar chart of a value aggregated by a category AND split by a second (sub)category. "
        "mode='grouped' compares side by side; mode='stacked' shows composition. Keep the breakdown to ≤6 values.",
        {
            "category_col": {"type": "string"},
            "breakdown_col": {"type": "string", "description": "Second categorical to split each bar by (≤6 values)."},
            "value_col": {"type": "string"},
            "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max"]},
            "mode": {"type": "string", "enum": ["grouped", "stacked"]},
            **_Y_FORMAT,
            **_TITLE_INTENT,
        },
        ["category_col", "breakdown_col", "value_col", "agg", "mode", "title", "intent"],
    ),
    _t(
        "dual_axis_chart",
        "Combo chart: a bar metric and a line metric on two y-axes, sharing an x category/time. "
        "Use when two metrics on different scales are worth seeing together (e.g. revenue + conversion rate).",
        {
            "x_col": {"type": "string"},
            "bar_value_col": {"type": "string"},
            "line_value_col": {"type": "string"},
            "bar_agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count"]},
            "line_agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count"]},
            **_Y_FORMAT,
            **_TITLE_INTENT,
        },
        ["x_col", "bar_value_col", "line_value_col", "bar_agg", "line_agg", "title", "intent"],
    ),
    _t(
        "treemap_chart",
        "Treemap of a category's share of a total (optionally a 2-level hierarchy). "
        "Prefer over a pie when there are many categories (>8) or a sub-category breakdown.",
        {
            "category_col": {"type": "string"},
            "subcategory_col": {"type": "string", "description": "Optional second level."},
            "value_col": {"type": "string"},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"]},
            **_Y_FORMAT,
            **_TITLE_INTENT,
        },
        ["category_col", "value_col", "agg", "title", "intent"],
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
