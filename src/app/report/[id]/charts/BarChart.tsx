'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const BAR_FILL = '#0D9488';      // teal-600 — confident, less mint than before
const TEXT_COLOR = '#44403C';    // stone-700
const AXIS_COLOR = '#A8A29E';    // stone-400

export default function BarChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const xData: any[] = spec.x ?? [];
  const yData: any[] = spec.y ?? [];

  const option = {
    textStyle: { color: TEXT_COLOR, fontFamily: 'inherit' },
    tooltip: {
      trigger: 'item',
      borderColor: '#E7E5E4',
      backgroundColor: '#ffffff',
      textStyle: { color: TEXT_COLOR, fontSize: 12 },
      formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)}`,
    },
    grid: { left: 70, right: 24, top: 24, bottom: xData.length > 10 ? 90 : 70 },
    xAxis: {
      type: 'category',
      data: xData,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: xData.length > 10 ? 60 : 40,
      nameTextStyle: { color: AXIS_COLOR, fontSize: 11 },
      axisLine: { lineStyle: { color: '#E7E5E4' } },
      axisTick: { show: false },
      axisLabel: {
        color: TEXT_COLOR,
        interval: 0,
        rotate: xData.length > 10 ? 30 : 0,
        fontSize: xData.length > 15 ? 10 : 11,
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
      splitLine: { lineStyle: { color: '#F5F5F4', type: 'solid' } },
      axisLabel: { color: AXIS_COLOR, formatter: fmtY, fontSize: 11 },
    },
    series: [{
      type: 'bar',
      data: yData,
      itemStyle: { color: BAR_FILL, borderRadius: [4, 4, 0, 0] },
      barMaxWidth: 56,
      label: {
        show: true,
        position: 'top',
        formatter: (p: any) => fmtY(p.value),
        color: TEXT_COLOR,
        fontSize: 11,
      },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 320 }} />;
}
