'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, monoFamily, CHART_PALETTE } from './chartTheme';

export default function Treemap({ spec }: { spec: any }) {
  const fmtV = getFormatter(spec.y_display_type);

  if (!spec.nodes?.length) {
    return <p className="text-sm text-ink-3">No data to display.</p>;
  }

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        tooltip: {
          ...chartBase().tooltip,
          formatter: (p: any) => `<strong>${p.name}</strong><br/>${fmtV(p.value)}`,
        },
        series: [{
          type: 'treemap',
          data: spec.nodes,
          roam: false,
          breadcrumb: { show: false },
          label: {
            show: true,
            fontFamily: monoFamily(),
            color: '#fff',
            formatter: '{b}',
          },
          itemStyle: { borderColor: '#fff', borderWidth: 2, gapWidth: 2 },
          levels: [
            { itemStyle: { borderColor: '#fff', borderWidth: 2, gapWidth: 2 } },
            { itemStyle: { borderColor: '#fff', borderWidth: 1, gapWidth: 1 } },
          ],
          color: CHART_PALETTE,
        }],
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
