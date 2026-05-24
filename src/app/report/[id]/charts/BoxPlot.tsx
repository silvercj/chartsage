'use client';
import ReactECharts from 'echarts-for-react';

export default function BoxPlot({ spec }: { spec: any }) {
  const series = spec.series ?? [];
  const categories = series.map((s: any) => s.name);
  const boxData = series.map((s: any) => [s.min, s.q1, s.median, s.q3, s.max]);
  const outliers = series.flatMap((s: any, i: number) => (s.outliers ?? []).map((v: number) => [i, v]));
  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'item' },
        grid: { left: 80, right: 40, top: 30, bottom: 60 },
        xAxis: { type: 'category', data: categories, name: spec.x_label, nameLocation: 'middle', nameGap: 30 },
        yAxis: { type: 'value', name: spec.y_label, nameLocation: 'middle', nameGap: 50 },
        series: [
          { type: 'boxplot', data: boxData, itemStyle: { color: '#73C0DE' } },
          { type: 'scatter', data: outliers, symbolSize: 6, itemStyle: { color: '#EE6666' } },
        ],
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
