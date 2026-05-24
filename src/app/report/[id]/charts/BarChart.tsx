'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

export default function BarChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const option = {
    tooltip: { trigger: 'item', formatter: (p: any) => `${p.name}: ${fmtY(p.value)}` },
    grid: { left: 80, right: 40, top: 30, bottom: 80 },
    xAxis: {
      type: 'category',
      data: spec.x,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: {
        interval: 0,
        rotate: spec.x.length > 10 ? 45 : 0,
        fontSize: spec.x.length > 15 ? 10 : 12,
      },
    },
    yAxis: {
      type: 'value',
      name: spec.y_label,
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: { formatter: fmtY },
    },
    series: [{
      type: 'bar',
      data: spec.y,
      itemStyle: { color: '#4ECDC4', borderRadius: [4, 4, 0, 0] },
      label: { show: true, position: 'top', formatter: (p: any) => fmtY(p.value) },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 360 }} />;
}
