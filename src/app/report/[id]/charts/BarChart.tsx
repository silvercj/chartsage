'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_TEAL, CHART_INK, CHART_INK_MUTED } from './chartTheme';

export default function BarChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const xData: any[] = spec.x ?? [];
  const yData: any[] = spec.y ?? [];

  const option = {
    ...chartBase(),
    tooltip: {
      ...chartBase().tooltip,
      trigger: 'item',
      formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)}`,
    },
    grid: { left: 8, right: 18, top: 24, bottom: xData.length > 10 ? 24 : 8, containLabel: true },
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
    series: [{
      type: 'bar',
      data: yData,
      itemStyle: { color: CHART_TEAL, borderRadius: [5, 5, 0, 0] },
      barWidth: '52%',
      emphasis: { itemStyle: { color: '#0A4A42' } },
      label: {
        show: true,
        position: 'top',
        formatter: (p: any) => fmtY(p.value),
        color: CHART_INK,
        fontSize: 11,
      },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 320 }} />;
}
