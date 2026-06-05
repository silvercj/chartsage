'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_TEAL, CHART_INK, CHART_INK_MUTED } from './chartTheme';
import { isWideChart } from './isWideChart';

const COLLAPSED_LIMIT = 12;

export default function BarChart({ spec, collapsed = false }: { spec: any; collapsed?: boolean }) {
  const fmtY = getFormatter(spec.y_display_type);

  // Wide (many-category) bar charts are horizontal rankings. Collapsed trims to
  // the top-12 in a standard-height card; expanded shows them all, full-width.
  const wide = isWideChart(spec);
  const horizontal = wide;
  const truncated = wide && collapsed;
  const allX: any[] = spec.x ?? [];
  const allY: any[] = spec.y ?? [];
  const xData = truncated ? allX.slice(0, COLLAPSED_LIMIT) : allX;
  const yData = truncated ? allY.slice(0, COLLAPSED_LIMIT) : allY;

  const valueAxis = valAxis({
    name: spec.y_label,
    nameLocation: 'middle',
    nameGap: horizontal ? 30 : 56,
    nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
    axisLabel: { formatter: fmtY },
  });

  const categoryAxis = catAxis({
    data: xData,
    inverse: horizontal,                       // highest-ranked bar at the top
    name: horizontal ? undefined : spec.x_label,
    nameLocation: 'middle',
    nameGap: xData.length > 10 ? 60 : 40,
    nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
    axisLabel: {
      interval: 0,
      rotate: horizontal ? 0 : (xData.length > 10 ? 30 : 0),
      fontSize: xData.length > 15 ? 10 : 11,
    },
  });

  const option = {
    ...chartBase(),
    tooltip: {
      ...chartBase().tooltip,
      trigger: 'item',
      formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)}`,
    },
    grid: horizontal
      ? { left: 8, right: 56, top: 8, bottom: 28, containLabel: true }
      : { left: 8, right: 18, top: 24, bottom: xData.length > 10 ? 24 : 8, containLabel: true },
    xAxis: horizontal ? valueAxis : categoryAxis,
    yAxis: horizontal ? categoryAxis : valueAxis,
    series: [{
      type: 'bar',
      data: yData,
      itemStyle: { color: CHART_TEAL, borderRadius: horizontal ? [0, 5, 5, 0] : [5, 5, 0, 0] },
      barWidth: '62%',
      emphasis: { itemStyle: { color: '#0A4A42' } },
      label: {
        show: true,
        position: horizontal ? 'right' : 'top',
        formatter: (p: any) => fmtY(p.value),
        color: CHART_INK,
        fontSize: 11,
      },
    }],
  };

  // Expanded ranking grows with row count (full-width, own row, bounded);
  // collapsed top-12 and normal charts stay the standard 320px.
  const height = (horizontal && !truncated)
    ? Math.min(760, Math.max(360, xData.length * 24 + 48))
    : 320;
  return <ReactECharts option={option} style={{ width: '100%', height }} />;
}
