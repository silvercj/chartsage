'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

// Calm, harmonious palette — earthy/professional, not Crayola.
const PIE_COLORS = [
  '#0D9488', '#7C3AED', '#F59E0B', '#0EA5E9',
  '#EF4444', '#10B981', '#EC4899', '#84CC16', '#A8A29E',
];
const TEXT_COLOR = '#44403C';

export default function PieChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const data = spec.x.map((name: string, i: number) => ({
    name,
    value: spec.y[i],
    itemStyle: { color: PIE_COLORS[i % PIE_COLORS.length] },
  }));

  return (
    <ReactECharts
      option={{
        textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
        tooltip: {
          trigger: 'item',
          borderColor: '#E7E5E4',
          backgroundColor: '#ffffff',
          textStyle: { color: TEXT_COLOR, fontSize: 12 },
          formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)} · ${p.percent}%`,
        },
        legend: { bottom: 0, textStyle: { color: TEXT_COLOR, fontSize: 11 }, icon: 'circle' },
        series: [{
          type: 'pie',
          radius: ['45%', '70%'],
          center: ['50%', '46%'],
          data,
          label: {
            formatter: '{b}\n{d}%',
            color: TEXT_COLOR,
            fontSize: 11,
            lineHeight: 14,
          },
          labelLine: { lineStyle: { color: '#A8A29E' } },
          itemStyle: { borderColor: '#ffffff', borderWidth: 2 },
        }],
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
