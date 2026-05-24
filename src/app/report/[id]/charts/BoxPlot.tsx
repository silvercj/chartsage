'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const BOX_FILL = '#0EA5E9';     // sky-500
const OUTLIER_FILL = '#EF4444'; // red-500
const TEXT_COLOR = '#44403C';
const AXIS_COLOR = '#A8A29E';

export default function BoxPlot({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const series = spec.series ?? [];
  const categories = series.map((s: any) => s.name);
  const boxData = series.map((s: any) => [s.min, s.q1, s.median, s.q3, s.max]);
  const outliers = series.flatMap((s: any, i: number) =>
    (s.outliers ?? []).map((v: number) => [i, v]),
  );

  return (
    <ReactECharts
      option={{
        textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
        tooltip: {
          trigger: 'item',
          borderColor: '#E7E5E4',
          backgroundColor: '#ffffff',
          textStyle: { color: TEXT_COLOR, fontSize: 12 },
          formatter: (p: any) => {
            if (Array.isArray(p.value) && p.value.length === 6) {
              const [, lo, q1, med, q3, hi] = p.value;
              return `<strong>${p.name}</strong><br/>` +
                `max: ${fmtY(hi)}<br/>q3: ${fmtY(q3)}<br/>` +
                `median: ${fmtY(med)}<br/>q1: ${fmtY(q1)}<br/>min: ${fmtY(lo)}`;
            }
            return `${fmtY(p.value[1])}`;
          },
        },
        grid: { left: 70, right: 24, top: 24, bottom: 48 },
        xAxis: {
          type: 'category',
          data: categories,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 30,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { lineStyle: { color: '#E7E5E4' } },
          axisTick: { show: false },
          axisLabel: { color: TEXT_COLOR, fontSize: 11 },
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
        series: [
          {
            type: 'boxplot',
            data: boxData,
            itemStyle: { color: BOX_FILL, borderColor: '#0369A1', borderWidth: 1.5 },
          },
          {
            type: 'scatter',
            data: outliers,
            symbolSize: 6,
            itemStyle: { color: OUTLIER_FILL, opacity: 0.6 },
          },
        ],
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
