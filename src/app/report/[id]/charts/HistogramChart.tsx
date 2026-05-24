'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

export default function HistogramChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const option = {
    tooltip: { trigger: 'item', formatter: (p: any) => `${p.name}: ${fmtY(p.value)}` },
    grid: { left: 80, right: 40, top: 30, bottom: 100 },
    xAxis: {
      type: 'category',
      data: spec.x,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: 70,
      axisLabel: { interval: 0, rotate: 45, fontSize: 10 },
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
      barCategoryGap: '0%',
      itemStyle: { color: '#5470C6' },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 360 }} />;
}
