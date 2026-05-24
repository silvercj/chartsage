'use client';
import ReactECharts from 'echarts-for-react';

const COLORS = ['#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272', '#FC8452', '#9A60B4'];

export default function ScatterChart({ spec }: { spec: any }) {
  const hasSeries = spec.series && Array.isArray(spec.series);
  const series = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'scatter',
        data: s.x.map((x: number, idx: number) => [x, s.y[idx]]),
        itemStyle: { color: COLORS[i % COLORS.length] },
      }))
    : [{
        type: 'scatter',
        data: spec.x.map((x: number, i: number) => [x, spec.y[i]]),
        itemStyle: { color: '#5470C6' },
      }];

  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'item' },
        legend: hasSeries ? { bottom: 0 } : undefined,
        grid: { left: 80, right: 40, top: 30, bottom: hasSeries ? 60 : 50 },
        xAxis: { type: 'value', name: spec.x_label, nameLocation: 'middle', nameGap: 30 },
        yAxis: { type: 'value', name: spec.y_label, nameLocation: 'middle', nameGap: 50 },
        series,
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
