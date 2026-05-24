'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const FILL = '#7C3AED';          // violet-600 — visually distinct from BarChart's teal
const TEXT_COLOR = '#44403C';
const AXIS_COLOR = '#A8A29E';

export default function HistogramChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const xData: any[] = spec.x ?? [];

  return (
    <ReactECharts
      option={{
        textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
        tooltip: {
          trigger: 'item',
          borderColor: '#E7E5E4',
          backgroundColor: '#ffffff',
          textStyle: { color: TEXT_COLOR, fontSize: 12 },
          formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)}`,
        },
        grid: { left: 70, right: 24, top: 24, bottom: 90 },
        xAxis: {
          type: 'category',
          data: xData,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 60,
          nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
          axisLine: { lineStyle: { color: '#E7E5E4' } },
          axisTick: { show: false },
          axisLabel: {
            color: AXIS_COLOR,
            interval: xData.length > 12 ? Math.max(1, Math.floor(xData.length / 8)) : 0,
            rotate: 35,
            fontSize: 10,
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
        series: [{
          type: 'bar',
          data: spec.y,
          barCategoryGap: '5%',
          itemStyle: { color: FILL, borderRadius: [2, 2, 0, 0] },
        }],
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
