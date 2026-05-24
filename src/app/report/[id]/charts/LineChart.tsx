'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const COLORS = ['#0D9488', '#7C3AED', '#F59E0B', '#EF4444', '#0EA5E9', '#10B981'];
const TEXT_COLOR = '#44403C';
const AXIS_COLOR = '#A8A29E';

function rollingAvg(values: number[], window: number): (number | null)[] {
  if (values.length < window || window <= 1) return values.map(() => null);
  const out: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    if (i < window - 1) {
      out.push(null);
      continue;
    }
    let sum = 0;
    for (let j = 0; j < window; j++) sum += values[i - j];
    out.push(sum / window);
  }
  return out;
}

export default function LineChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const hasSeries = spec.series && Array.isArray(spec.series);
  const xData = hasSeries ? spec.series[0].x : spec.x;
  const xLen = xData?.length ?? 0;
  const showSmoothed = xLen >= 12 && !hasSeries;   // only on single-series, long enough to be useful

  const baseSeries = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'line',
        data: s.y,
        smooth: 0.3,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { width: 2 },
        itemStyle: { color: COLORS[i % COLORS.length] },
      }))
    : [{
        name: spec.y_label || 'value',
        type: 'line',
        data: spec.y,
        smooth: 0.3,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { width: 2 },
        itemStyle: { color: COLORS[0] },
      }];

  if (showSmoothed) {
    const smoothed = rollingAvg(spec.y as number[], 3);
    baseSeries.push({
      name: '3-mo avg',
      type: 'line',
      data: smoothed,
      smooth: 0.5,
      symbol: 'none',
      lineStyle: { width: 2, color: '#1C1917', type: 'dashed', opacity: 0.55 },
      itemStyle: { color: '#1C1917' },
    } as any);
  }

  const showLegend = hasSeries || showSmoothed;

  return (
    <ReactECharts
      option={{
        textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
        tooltip: {
          trigger: 'axis',
          borderColor: '#E7E5E4',
          backgroundColor: '#ffffff',
          textStyle: { color: TEXT_COLOR, fontSize: 12 },
          formatter: (p: any[]) =>
            p
              .filter((x) => x.value !== null && x.value !== undefined)
              .map((x) => `<span style="color:${x.color}">●</span> ${x.seriesName ?? ''}: ${fmtY(x.value)}`)
              .join('<br/>'),
        },
        legend: showLegend
          ? {
              bottom: 0,
              textStyle: { color: TEXT_COLOR, fontSize: 11 },
              itemWidth: 18,
              itemHeight: 2,
              itemGap: 20,
            }
          : undefined,
        grid: { left: 70, right: 24, top: 24, bottom: showLegend ? 84 : 48 },
        xAxis: {
          type: 'category',
          data: xData,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 34,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { lineStyle: { color: '#E7E5E4' } },
          axisTick: { show: false },
          axisLabel: {
            color: AXIS_COLOR,
            fontSize: 11,
            interval: xLen > 24 ? Math.floor(xLen / 12) : 'auto',
          },
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
          axisLabel: { color: AXIS_COLOR, formatter: fmtY, fontSize: 11 },
        },
        series: baseSeries,
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
