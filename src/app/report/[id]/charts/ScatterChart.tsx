'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const COLORS = ['#0D9488', '#7C3AED', '#F59E0B', '#EF4444', '#0EA5E9', '#10B981', '#EC4899', '#84CC16'];
const TEXT_COLOR = '#44403C';
const AXIS_COLOR = '#A8A29E';

export default function ScatterChart({ spec }: { spec: any }) {
  const fmtX = getFormatter(spec.x_display_type === 'number' ? 'number' : undefined);
  const fmtY = getFormatter(spec.y_display_type);
  const hasSeries = spec.series && Array.isArray(spec.series);

  const series = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'scatter',
        data: s.x.map((x: number, idx: number) => [x, s.y[idx]]),
        itemStyle: { color: COLORS[i % COLORS.length], opacity: 0.7 },
        symbolSize: 7,
      }))
    : [{
        type: 'scatter',
        data: spec.x.map((x: number, i: number) => [x, spec.y[i]]),
        itemStyle: { color: COLORS[0], opacity: 0.65 },
        symbolSize: 7,
      }];

  return (
    <ReactECharts
      option={{
        textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
        tooltip: {
          trigger: 'item',
          borderColor: '#E7E5E4',
          backgroundColor: '#ffffff',
          textStyle: { color: TEXT_COLOR, fontSize: 12 },
          formatter: (p: any) =>
            `${spec.x_label}: ${fmtX(p.value[0])}<br/>${spec.y_label}: ${fmtY(p.value[1])}`,
        },
        legend: hasSeries
          ? { bottom: 0, textStyle: { color: TEXT_COLOR, fontSize: 11 }, icon: 'circle' }
          : undefined,
        grid: { left: 70, right: 24, top: 24, bottom: hasSeries ? 56 : 40 },
        xAxis: {
          type: 'value',
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 30,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { lineStyle: { color: '#E7E5E4' } },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: '#F5F5F4' } },
          axisLabel: { color: AXIS_COLOR, fontSize: 11, formatter: fmtX },
        },
        yAxis: {
          type: 'value',
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 56,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: '#F5F5F4' } },
          axisLabel: { color: AXIS_COLOR, fontSize: 11, formatter: fmtY },
        },
        series,
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
