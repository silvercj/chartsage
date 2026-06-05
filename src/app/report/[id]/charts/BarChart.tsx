'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_TEAL, CHART_INK, CHART_INK_MUTED } from './chartTheme';
import { isWideChart } from './isWideChart';

const COLLAPSED_LIMIT = 12;

export default function BarChart({ spec, collapsed = false }: { spec: any; collapsed?: boolean }) {
  const fmtY = getFormatter(spec.y_display_type);

  // Bars stay vertical at a fixed height — never tall. A wide chart grows
  // SIDEWAYS instead: expanded it spans both columns (handled by the card) so
  // all bars get horizontal room; collapsed it trims to the top-12 in a normal
  // card. Per-bar value labels collide past ~a dozen, so only show them then.
  const wide = isWideChart(spec);
  const limited = wide && collapsed;
  const xData: any[] = limited ? (spec.x ?? []).slice(0, COLLAPSED_LIMIT) : (spec.x ?? []);
  const yData: any[] = limited ? (spec.y ?? []).slice(0, COLLAPSED_LIMIT) : (spec.y ?? []);
  const n = xData.length;

  const option = {
    ...chartBase(),
    tooltip: {
      ...chartBase().tooltip,
      trigger: 'item',
      formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtY(p.value)}`,
    },
    grid: { left: 8, right: 18, top: 24, bottom: n > 10 ? 24 : 8, containLabel: true },
    xAxis: catAxis({
      data: xData,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: n > 10 ? 60 : 40,
      nameTextStyle: { color: CHART_INK_MUTED, fontSize: 11 },
      axisLabel: {
        interval: 0,
        rotate: n > 10 ? 30 : 0,
        fontSize: n > 15 ? 10 : 11,
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
      data: yData,
      itemStyle: { color: CHART_TEAL, borderRadius: [5, 5, 0, 0] },
      barWidth: '62%',
      emphasis: { itemStyle: { color: '#0A4A42' } },
      label: {
        show: true,
        position: 'top',
        formatter: (p: any) => fmtY(p.value),
        color: CHART_INK,
        fontSize: 11,
      },
      // Keep value labels, but let ECharts drop only the ones that would
      // overlap: a full-width chart has room to show them; a dense/small one
      // hides the colliders instead of smearing.
      labelLayout: { hideOverlap: true },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 320 }} />;
}
