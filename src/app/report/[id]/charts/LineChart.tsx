'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_PALETTE, CHART_INK, CHART_INK_MUTED, tealAreaGradient } from './chartTheme';

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
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        showSymbol: false,
        lineStyle: { width: 2.5 },
        itemStyle: { color: CHART_PALETTE[i % CHART_PALETTE.length] },
        emphasis: { showSymbol: true },
      }))
    : [{
        name: spec.y_label || 'value',
        type: 'line',
        data: spec.y,
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        showSymbol: false,
        lineStyle: { width: 2.5, color: CHART_PALETTE[0] },
        itemStyle: { color: CHART_PALETTE[0] },
        emphasis: { showSymbol: true },
        areaStyle: { color: tealAreaGradient() },
      }];

  if (showSmoothed) {
    const smoothed = rollingAvg(spec.y as number[], 3);
    baseSeries.push({
      name: '3-mo avg',
      type: 'line',
      data: smoothed,
      smooth: true,
      symbol: 'none',
      lineStyle: { width: 2, color: CHART_INK, type: 'dashed', opacity: 0.5 },
      itemStyle: { color: CHART_INK },
    } as any);
  }

  const showLegend = hasSeries || showSmoothed;

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        grid: { left: 8, right: 18, top: 24, bottom: showLegend ? 40 : 8, containLabel: true },
        tooltip: {
          ...chartBase().tooltip,
          trigger: 'axis',
          formatter: (p: any[]) =>
            p
              .filter((x) => x.value !== null && x.value !== undefined)
              .map((x) => `<span style="color:${x.color}">●</span> ${x.seriesName ?? ''}: ${fmtY(x.value)}`)
              .join('<br/>'),
        },
        legend: showLegend
          ? {
              bottom: 0,
              textStyle: { fontFamily: chartBase().textStyle.fontFamily, fontSize: 11, color: CHART_INK },
              icon: 'roundRect',
              itemWidth: 10,
              itemHeight: 10,
              itemGap: 20,
            }
          : undefined,
        xAxis: catAxis({
          data: xData,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 34,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: {
            interval: xLen > 24 ? Math.floor(xLen / 12) : 'auto',
            rotate: xLen > 8 ? 30 : 0,
          },
        }),
        yAxis: valAxis({
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 56,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: { formatter: fmtY },
        }),
        series: baseSeries,
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
