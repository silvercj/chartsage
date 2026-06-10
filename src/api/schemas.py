"""Pydantic models shared across the backend."""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


ChartKind = Literal["bar", "histogram", "scatter", "line", "pie", "box", "heatmap", "grouped_bar", "dual_axis", "treemap"]
LayoutPosition = Literal["main", "sidebar"]
ColumnRole = Literal["categorical", "numeric", "date", "identifier", "unusable"]
XDisplayType = Literal["category", "number", "date", "text"]
YDisplayType = Literal["count", "currency", "percentage", "number"]


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: ColumnRole
    cardinality: int
    null_count: int
    top_values: Optional[list[tuple[Any, int]]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    unusable_reason: Optional[str] = None
    multi_value: bool = False
    delimiter: Optional[str] = None
    # Numeric column that's really a time/ordinal axis (Year, Decade, Quarter — one row
    # per period). Detected in profile.py; consumed by the prompt text, the fallback's
    # lead-line pick, and the key-metrics role guard.
    temporal_ordinal: bool = False


class DataProfile(BaseModel):
    row_count: int
    columns: list[ColumnInfo]
    correlations: dict[str, float] = Field(default_factory=dict)  # key = "col1||col2"
    anomalies: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        """Compact text form for sending to Claude."""
        lines = [f"Rows: {self.row_count}", f"Columns ({len(self.columns)}):"]
        for c in self.columns:
            parts = [f"- {c.name}: role={c.role}, dtype={c.dtype}, cardinality={c.cardinality}, nulls={c.null_count}"]
            if c.role == "numeric":
                parts.append(f"  min={c.min}, max={c.max}, mean={c.mean}, median={c.median}, std={c.std}")
                if c.temporal_ordinal:
                    parts.append(
                        "  ordinal time axis (one row per period) — chart metrics OVER it "
                        "(line_chart date_col, or the x of a comparison); never histogram it, "
                        "aggregate it as a metric, or report it as a key metric"
                    )
            elif c.role == "categorical" and c.top_values:
                top = ", ".join(f"{v}={n}" for v, n in c.top_values[:5])
                if c.multi_value:
                    parts.append(f"  multi-value (split on '{c.delimiter}') — top: {top}")
                else:
                    parts.append(f"  top: {top}")
            elif c.role == "date":
                parts.append(f"  range: {c.min_date} → {c.max_date}")
            elif c.role == "unusable":
                parts.append(f"  unusable_reason: {c.unusable_reason}")
            lines.extend(parts)
        if self.correlations:
            lines.append("Correlations (|r| ≥ 0.3):")
            for pair, r in self.correlations.items():
                lines.append(f"- {pair}: {r:.2f}")
        if self.anomalies:
            lines.append("Anomalies:")
            for a in self.anomalies:
                lines.append(f"- {a}")
        return "\n".join(lines)


class ChartSpec(BaseModel):
    kind: ChartKind
    title: str
    intent: str
    x: Optional[list[Any]] = None
    y: Optional[list[Any]] = None
    series: Optional[list[dict]] = None
    x_label: str = ""
    y_label: str = ""
    x_display_type: XDisplayType = "category"
    y_display_type: YDisplayType = "number"
    stacked: bool = False
    area: bool = False
    y_label_secondary: Optional[str] = None
    nodes: Optional[list] = None
    source_columns: list[str]
    data_point_count: int


class ToolError(BaseModel):
    reason: str


KpiFormat = Literal["number", "currency", "percent"]


class KeyMetric(BaseModel):
    label: str
    value: float
    format: KpiFormat = "number"


class ChartLayoutEntry(BaseModel):
    chart_id: str
    position: LayoutPosition
    order: int
    collapsed: bool = False   # wide bar charts: owner-collapsed to the compact top-N view


class ChartWithCaption(BaseModel):
    chart_id: str
    spec: ChartSpec
    caption: str


class ReportNarrative(BaseModel):
    summary: str
    captions: list[str]
    data_quality: list[str]


class Report(BaseModel):
    generated_at: str
    summary: str
    data_quality: list[str]
    key_metrics: list[KeyMetric] = Field(default_factory=list)
    charts: list[ChartWithCaption]
    layout: list[ChartLayoutEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
