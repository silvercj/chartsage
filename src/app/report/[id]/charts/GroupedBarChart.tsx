'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_PALETTE, CHART_INK, CHART_INK_MUTED, monoFamily } from './chartTheme';

export default function GroupedBarChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const xData: any[] = spec.x ?? [];
  const seriesSpecs: any[] = Array.isArray(spec.series) ? spec.series : [];
  const stacked = !!spec.stacked;

  const series = seriesSpecs.map((s: any, i: number) => ({
    name: s.name,
    type: 'bar',
    data: s.data,
    ...(stacked ? { stack: 'total' } : {}),
    itemStyle: { color: CHART_PALETTE[i % CHART_PALETTE.length], borderRadius: [4, 4, 0, 0] },
  }));

  const option = {
    ...chartBase(),
    tooltip: {
      ...chartBase().tooltip,
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (p: any[]) =>
        p
          .filter((x) => x.value !== null && x.value !== undefined)
          .map((x) => `<span style="color:${x.color}">●</span> ${x.seriesName ?? ''}: ${fmtY(x.value)}`)
          .join('<br/>'),
    },
    legend: {
      bottom: 0,
      textStyle: { fontFamily: monoFamily(), fontSize: 11, color: CHART_INK },
      icon: 'roundRect',
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 20,
    },
    grid: { left: 8, right: 18, top: 24, bottom: 40, containLabel: true },
    xAxis: catAxis({
      data: xData,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: xData.length > 10 ? 60 : 40,
      nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
      axisLabel: {
        interval: 0,
        rotate: xData.length > 10 ? 30 : 0,
        fontSize: xData.length > 15 ? 10 : 11,
      },
    }),
    yAxis: valAxis({
      name: spec.y_label,
      nameLocation: 'middle',
      nameGap: 56,
      nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
      axisLabel: { formatter: fmtY },
    }),
    series,
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 320 }} />;
}
