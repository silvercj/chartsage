'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_TEAL, CHART_INK_MUTED } from './chartTheme';

export default function HistogramChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const xData: any[] = spec.x ?? [];

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        tooltip: {
          ...chartBase().tooltip,
          trigger: 'item',
          formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)}`,
        },
        grid: { left: 8, right: 18, top: 24, bottom: 24, containLabel: true },
        xAxis: catAxis({
          data: xData,
          name: spec.x_label,
          nameLocation: 'middle',
          nameGap: 60,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: {
            interval: xData.length > 12 ? Math.max(1, Math.floor(xData.length / 8)) : 0,
            rotate: 35,
            fontSize: 10,
          },
        }),
        yAxis: valAxis({
          name: spec.y_label,
          nameLocation: 'middle',
          nameGap: 56,
          nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
          axisLabel: { formatter: fmtY },
        }),
        series: [{
          type: 'bar',
          data: spec.y,
          barCategoryGap: '5%',
          itemStyle: { color: CHART_TEAL, borderRadius: [5, 5, 0, 0] },
          emphasis: { itemStyle: { color: '#0A4A42' } },
        }],
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
