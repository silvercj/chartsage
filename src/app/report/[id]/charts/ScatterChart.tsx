'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, valAxis, CHART_PALETTE, CHART_TEAL, CHART_INK, CHART_INK_MUTED } from './chartTheme';

export default function ScatterChart({ spec }: { spec: any }) {
  const fmtX = getFormatter(spec.x_display_type === 'number' ? 'number' : undefined);
  const fmtY = getFormatter(spec.y_display_type);
  const hasSeries = spec.series && Array.isArray(spec.series);

  const series = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'scatter',
        data: s.x.map((x: number, idx: number) => [x, s.y[idx]]),
        itemStyle: { color: CHART_PALETTE[i % CHART_PALETTE.length], opacity: 0.55 },
        symbolSize: 7,
      }))
    : [{
        type: 'scatter',
        data: spec.x.map((x: number, i: number) => [x, spec.y[i]]),
        itemStyle: { color: CHART_TEAL, opacity: 0.55 },
        symbolSize: 7,
      }];

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        tooltip: {
          ...chartBase().tooltip,
          trigger: 'item',
          formatter: (p: any) =>
            `${spec.x_label}: ${fmtX(p.value[0])}<br/>${spec.y_label}: ${fmtY(p.value[1])}`,
        },
        legend: hasSeries
          ? {
              bottom: 0,
              textStyle: { fontFamily: chartBase().textStyle.fontFamily, fontSize: 11, color: CHART_INK },
              icon: 'circle',
              itemWidth: 10,
              itemHeight: 10,
              itemGap: 18,
            }
          : undefined,
        grid: { left: 8, right: 18, top: 24, bottom: hasSeries ? 40 : 8, containLabel: true },
        xAxis: valAxis({
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 32,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: { formatter: fmtX },
        }),
        yAxis: valAxis({
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 56,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: { formatter: fmtY },
        }),
        series,
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
