'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

export default function PieChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const data = spec.x.map((name: string, i: number) => ({ name, value: spec.y[i] }));
  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'item', formatter: (p: any) => `${p.name}: ${fmtY(p.value)} (${p.percent}%)` },
        legend: { bottom: 0 },
        series: [{
          type: 'pie',
          radius: ['30%', '60%'],
          data,
          label: { formatter: '{b}: {d}%' },
        }],
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
