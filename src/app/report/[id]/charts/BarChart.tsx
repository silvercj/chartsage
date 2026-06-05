'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_TEAL, CHART_INK, CHART_INK_MUTED } from './chartTheme';

export default function BarChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const xData: any[] = spec.x ?? [];
  const yData: any[] = spec.y ?? [];

  // Many categories → horizontal ranked leaderboard. Vertical bars cram the
  // x-axis labels and pile value-labels on top of each other past ~a dozen bars;
  // horizontal puts names on readable rows and one value at each bar's end.
  const horizontal = xData.length > 12;

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

  // Horizontal charts need height proportional to the number of rows.
  const height = horizontal ? Math.max(360, xData.length * 24 + 48) : 320;
  return <ReactECharts option={option} style={{ width: '100%', height }} />;
}
