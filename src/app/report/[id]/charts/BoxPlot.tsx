'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_TEAL, CHART_INK_MUTED } from './chartTheme';

const OUTLIER_FILL = '#B5673A'; // clay — warm accent for outlier points

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
        ...chartBase(),
        tooltip: {
          ...chartBase().tooltip,
          trigger: 'item',
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
        grid: { left: 8, right: 18, top: 24, bottom: 8, containLabel: true },
        xAxis: catAxis({
          data: categories,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 30,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
        }),
        yAxis: valAxis({
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 56,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: { formatter: fmtY },
        }),
        series: [
          {
            type: 'boxplot',
            data: boxData,
            itemStyle: { color: 'rgba(12,92,82,0.12)', borderColor: CHART_TEAL, borderWidth: 1.5 },
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
