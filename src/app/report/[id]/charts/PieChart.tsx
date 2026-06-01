'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, monoFamily, CHART_PALETTE, CHART_INK, CHART_INK_MUTED } from './chartTheme';

export default function PieChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const data = spec.x.map((name: string, i: number) => ({
    name,
    value: spec.y[i],
  }));

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        color: CHART_PALETTE,
        tooltip: {
          ...chartBase().tooltip,
          trigger: 'item',
          formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)} · ${p.percent}%`,
        },
        legend: {
          bottom: 0,
          textStyle: { fontFamily: monoFamily(), fontSize: 11, color: CHART_INK },
          icon: 'circle',
        },
        series: [{
          type: 'pie',
          radius: ['45%', '70%'],
          center: ['50%', '46%'],
          data,
          label: {
            formatter: '{b}\n{d}%',
            fontFamily: monoFamily(),
            color: CHART_INK,
            fontSize: 11,
            lineHeight: 14,
          },
          labelLine: { lineStyle: { color: CHART_INK_MUTED } },
          itemStyle: { borderColor: '#fff', borderWidth: 2 },
        }],
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
