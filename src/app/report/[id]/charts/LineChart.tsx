'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const COLORS = ['#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272'];

export default function LineChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const hasSeries = spec.series && Array.isArray(spec.series);
  const xData = hasSeries ? spec.series[0].x : spec.x;
  const series = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'line',
        data: s.y,
        smooth: true,
        itemStyle: { color: COLORS[i % COLORS.length] },
      }))
    : [{ type: 'line', data: spec.y, smooth: true, itemStyle: { color: '#5470C6' } }];

  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'axis', formatter: (p: any[]) => p.map((x) => `${x.seriesName ?? ''}: ${fmtY(x.value)}`).join('<br/>') },
        legend: hasSeries ? { bottom: 0 } : undefined,
        grid: { left: 80, right: 40, top: 30, bottom: hasSeries ? 60 : 50 },
        xAxis: { type: 'category', data: xData, name: spec.x_label, nameLocation: 'middle', nameGap: 30 },
        yAxis: { type: 'value', name: spec.y_label, nameLocation: 'middle', nameGap: 50, axisLabel: { formatter: fmtY } },
        series,
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
