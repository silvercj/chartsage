from chart_tools import CHART_TOOLS, NARRATIVE_TOOL


def test_all_eight_tools_present():
    names = {t["name"] for t in CHART_TOOLS}
    assert names == {
        "frequency_bar_chart",
        "aggregation_bar_chart",
        "histogram_chart",
        "scatter_chart",
        "line_chart",
        "pie_chart",
        "box_plot",
        "heatmap_chart",
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


def test_narrative_tool_shape():
    assert NARRATIVE_TOOL["name"] == "submit_narrative"
    props = NARRATIVE_TOOL["input_schema"]["properties"]
    assert "summary" in props
    assert "captions" in props
    assert "data_quality" in props
