'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const TEXT_COLOR = '#44403C';
const AXIS_COLOR = '#A8A29E';

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

  return (
    <ReactECharts
      option={{
        textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
        tooltip: {
          position: 'top',
          borderColor: '#E7E5E4',
          backgroundColor: '#ffffff',
          textStyle: { color: TEXT_COLOR, fontSize: 12 },
          formatter: (p: any) =>
            `${yLabels[p.data[1]]} × ${xLabels[p.data[0]]}<br/><strong>${fmtV(p.data[2])}</strong>`,
        },
        grid: { left: 110, right: 24, top: 24, bottom: 70 },
        xAxis: {
          type: 'category',
          data: xLabels,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 50,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: TEXT_COLOR, rotate: 45, fontSize: 10, interval: 0 },
          splitArea: { show: false },
        },
        yAxis: {
          type: 'category',
          data: yLabels,
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 90,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: TEXT_COLOR, fontSize: 10, interval: 0 },
        },
        visualMap: {
          min: symmetric ? -Math.max(Math.abs(vMin), Math.abs(vMax)) : vMin,
          max: symmetric ? Math.max(Math.abs(vMin), Math.abs(vMax)) : vMax,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: 8,
          textStyle: { color: AXIS_COLOR, fontSize: 10 },
          itemWidth: 12,
          itemHeight: 100,
          inRange: {
            color: symmetric
              ? ['#EF4444', '#FFFFFF', '#0D9488']
              : ['#F5F5F4', '#0D9488'],
          },
        },
        series: [{
          type: 'heatmap',
          data,
          label: {
            show: !dense,
            formatter: (p: any) => fmtV(p.data[2]),
            fontSize: 10,
            color: TEXT_COLOR,
          },
          itemStyle: { borderColor: '#ffffff', borderWidth: 1 },
        }],
      }}
      style={{ width: '100%', height: 380 }}
    />
  );
}
