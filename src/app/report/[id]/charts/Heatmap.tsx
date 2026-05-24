'use client';
import ReactECharts from 'echarts-for-react';

export default function Heatmap({ spec }: { spec: any }) {
  const series = spec.series ?? [];
  const xLabels: string[] = spec.x ?? [];
  const yLabels: string[] = spec.y ?? [];
  const data = series.map((s: any) => [xLabels.indexOf(s.col), yLabels.indexOf(s.row), s.value]);
  const values = series.map((s: any) => s.value);
  const vMin = Math.min(...values, 0);
  const vMax = Math.max(...values, 0);
  return (
    <ReactECharts
      option={{
        tooltip: {
          position: 'top',
          formatter: (p: any) => `${yLabels[p.data[1]]} × ${xLabels[p.data[0]]}: ${p.data[2].toFixed(2)}`,
        },
        grid: { left: 100, right: 40, top: 30, bottom: 60 },
        xAxis: { type: 'category', data: xLabels, axisLabel: { rotate: 30 }, name: spec.x_label, nameLocation: 'middle', nameGap: 40 },
        yAxis: { type: 'category', data: yLabels, name: spec.y_label, nameLocation: 'middle', nameGap: 70 },
        visualMap: { min: vMin, max: vMax, calculable: true, orient: 'horizontal', left: 'center', bottom: 0 },
        series: [{
          type: 'heatmap',
          data,
          label: { show: true, formatter: (p: any) => p.data[2].toFixed(2), fontSize: 10 },
        }],
      }}
      style={{ width: '100%', height: 380 }}
    />
  );
}
