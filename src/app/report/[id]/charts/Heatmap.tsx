'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, monoFamily, CHART_INK_MUTED } from './chartTheme';

export default function Heatmap({ spec }: { spec: any }) {
  const fmtV = getFormatter(spec.y_display_type);
  const series = spec.series ?? [];
  const xLabels: string[] = spec.x ?? [];
  const yLabels: string[] = spec.y ?? [];
  const data = series.map((s: any) => [xLabels.indexOf(s.col), yLabels.indexOf(s.row), s.value]);
  const values = series.map((s: any) => s.value);
  const vMin = values.length ? Math.min(...values) : 0;
  const vMax = values.length ? Math.max(...values) : 1;
  const symmetric = vMin < 0 && vMax > 0;
  // Dense matrices (e.g. a 14×14 correlation grid) are unreadable with a number
  // crammed in every cell — show colour + hover value instead, keep labels only
  // when the grid is small enough for them to fit.
  const dense = data.length > 49;

  const mono = monoFamily();

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        tooltip: {
          ...chartBase().tooltip,
          position: 'top',
          formatter: (p: any) =>
            `${yLabels[p.data[1]]} × ${xLabels[p.data[0]]}<br/><strong>${fmtV(p.data[2])}</strong>`,
        },
        grid: { left: 110, right: 24, top: 24, bottom: 70, containLabel: true },
        xAxis: {
          type: 'category',
          data: xLabels,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 50,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { fontFamily: mono, color: CHART_INK_MUTED, rotate: 45, fontSize: 10, interval: 0 },
          splitArea: { show: false },
        },
        yAxis: {
          type: 'category',
          data: yLabels,
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 90,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { fontFamily: mono, color: CHART_INK_MUTED, fontSize: 10, interval: 0 },
        },
        visualMap: {
          min: symmetric ? -Math.max(Math.abs(vMin), Math.abs(vMax)) : vMin,
          max: symmetric ? Math.max(Math.abs(vMin), Math.abs(vMax)) : vMax,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: 8,
          textStyle: { fontFamily: mono, color: CHART_INK_MUTED, fontSize: 10 },
          itemWidth: 12,
          itemHeight: 100,
          inRange: {
            color: symmetric
              ? ['#B5673A', '#F6F3EC', '#0C5C52']
              : ['#F6F3EC', '#0C5C52'],
          },
        },
        series: [{
          type: 'heatmap',
          data,
          label: {
            show: !dense,
            formatter: (p: any) => fmtV(p.data[2]),
            fontFamily: mono,
            fontSize: 10,
            color: CHART_INK_MUTED,
          },
          itemStyle: { borderColor: '#fff', borderWidth: 1 },
        }],
      }}
      style={{ width: '100%', height: 380 }}
    />
  );
}
