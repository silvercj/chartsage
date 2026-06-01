from chart_tools import CHART_TOOLS, NARRATIVE_TOOL


def test_all_tools_present():
    names = {t["name"] for t in CHART_TOOLS}
    assert names == {
        "key_metrics",
        "frequency_bar_chart",
        "aggregation_bar_chart",
        "histogram_chart",
        "scatter_chart",
        "line_chart",
        "pie_chart",
        "box_plot",
        "heatmap_chart",
        "grouped_bar_chart",
    }


def test_each_tool_has_required_shape():
    for tool in CHART_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        # key_metrics is not a chart — it has no title/intent (it's routed to
        # Report.key_metrics, never captioned or laid out). Every chart tool does.
        if tool["name"] == "key_metrics":
            continue
        assert "title" in schema["properties"]
        assert "intent" in schema["properties"]
        assert "title" in schema["required"]
        assert "intent" in schema["required"]


def test_frequency_tool_signature():
    tool = next(t for t in CHART_TOOLS if t["name"] == "frequency_bar_chart")
    props = tool["input_schema"]["properties"]
    assert "column" in props
    assert tool["input_schema"]["required"] == ["column", "title", "intent"]


def test_aggregation_tool_excludes_count():
    tool = next(t for t in CHART_TOOLS if t["name"] == "aggregation_bar_chart")
    agg = tool["input_schema"]["properties"]["agg"]
    assert "count" not in agg["enum"]
    assert "sum" in agg["enum"]


def test_line_tool_includes_count():
    tool = next(t for t in CHART_TOOLS if t["name"] == "line_chart")
    agg = tool["input_schema"]["properties"]["agg"]
    assert "count" in agg["enum"]


def test_key_metrics_tool_shape():
    tool = next(t for t in CHART_TOOLS if t["name"] == "key_metrics")
    schema = tool["input_schema"]
    assert schema["required"] == ["metrics"]
    metrics = schema["properties"]["metrics"]
    assert metrics["type"] == "array"
    assert metrics["maxItems"] == 5
    item_props = metrics["items"]["properties"]
    assert set(metrics["items"]["required"]) == {"label", "column", "agg"}
    assert set(item_props["agg"]["enum"]) == {
        "sum", "mean", "median", "min", "max", "count", "nunique"}
    assert set(item_props["format"]["enum"]) == {"number", "currency", "percent"}
    # key_metrics is NOT a chart: no title/intent on the tool itself
    assert "title" not in schema["properties"]
    assert "intent" not in schema["properties"]


def test_narrative_tool_shape():
    assert NARRATIVE_TOOL["name"] == "submit_narrative"
    props = NARRATIVE_TOOL["input_schema"]["properties"]
    assert "summary" in props
    assert "captions" in props
    assert "data_quality" in props
