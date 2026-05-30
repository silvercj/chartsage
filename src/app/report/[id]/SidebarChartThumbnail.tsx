'use client';
import ReactECharts from 'echarts-for-react';

const ACCENT = '#0D9488';   // teal — bar/line/scatter
const VIOLET = '#7C3AED';   // histogram
const SKY = '#0EA5E9';      // box
const PIE_COLORS = ['#0D9488', '#7C3AED', '#F59E0B', '#0EA5E9', '#EF4444', '#10B981', '#EC4899', '#84CC16', '#A8A29E'];

const HIDDEN_AXIS = {
  show: false,
  axisLine: { show: false },
  axisTick: { show: false },
  axisLabel: { show: false },
  splitLine: { show: false },
};

const SMALL_GRID = { left: 3, right: 3, top: 4, bottom: 3, containLabel: false };

/** Build a minimal, label-free ECharts option that conveys the chart's shape at ~64px tall. */
function thumbnailOption(spec: any): any | null {
  const k = spec?.kind;
  const common = { animation: false, tooltip: { show: false } };

  if (k === 'bar' || k === 'histogram') {
    return {
      ...common,
      grid: SMALL_GRID,
      xAxis: { type: 'category', data: spec.x ?? [], ...HIDDEN_AXIS },
      yAxis: { type: 'value', ...HIDDEN_AXIS },
      series: [{
        type: 'bar',
        data: spec.y ?? [],
        itemStyle: { color: k === 'histogram' ? VIOLET : ACCENT, borderRadius: [1, 1, 0, 0] },
        barCategoryGap: k === 'histogram' ? '6%' : '22%',
      }],
    };
  }

  if (k === 'line') {
    const y = spec.series ? (spec.series[0]?.y ?? []) : (spec.y ?? []);
    const x = spec.series ? (spec.series[0]?.x ?? []) : (spec.x ?? []);
    return {
      ...common,
      grid: SMALL_GRID,
      xAxis: { type: 'category', data: x, ...HIDDEN_AXIS },
      yAxis: { type: 'value', ...HIDDEN_AXIS, scale: true },
      series: [{
        type: 'line', data: y, smooth: true, symbol: 'none',
        lineStyle: { width: 1.5, color: ACCENT },
        areaStyle: { color: 'rgba(13,148,136,0.12)' },
      }],
    };
  }

  if (k === 'scatter') {
    let pts: number[][] = [];
    if (spec.series) {
      for (const s of spec.series) for (let i = 0; i < (s.x?.length ?? 0); i++) pts.push([s.x[i], s.y[i]]);
    } else {
      const x = spec.x ?? [], y = spec.y ?? [];
      for (let i = 0; i < x.length; i++) pts.push([x[i], y[i]]);
    }
    if (pts.length > 80) {
      const step = Math.ceil(pts.length / 80);
      pts = pts.filter((_, i) => i % step === 0);
    }
    return {
      ...common,
      grid: SMALL_GRID,
      xAxis: { type: 'value', ...HIDDEN_AXIS, scale: true },
      yAxis: { type: 'value', ...HIDDEN_AXIS, scale: true },
      series: [{ type: 'scatter', data: pts, symbolSize: 2.5, itemStyle: { color: ACCENT, opacity: 0.55 } }],
    };
  }

  if (k === 'pie') {
    const data = (spec.x ?? []).map((name: string, i: number) => ({
      name, value: spec.y?.[i], itemStyle: { color: PIE_COLORS[i % PIE_COLORS.length] },
    }));
    return {
      ...common,
      series: [{
        type: 'pie', radius: ['45%', '85%'], center: ['50%', '50%'],
        label: { show: false }, labelLine: { show: false }, data,
        itemStyle: { borderColor: '#fff', borderWidth: 0.5 },
      }],
    };
  }

  if (k === 'box') {
    const series = spec.series ?? [];
    const boxData = series.map((s: any) => [s.min, s.q1, s.median, s.q3, s.max]);
    return {
      ...common,
      grid: SMALL_GRID,
      xAxis: { type: 'category', data: series.map((_: any, i: number) => i), ...HIDDEN_AXIS },
      yAxis: { type: 'value', ...HIDDEN_AXIS, scale: true },
      series: [{
        type: 'boxplot', data: boxData,
        itemStyle: { color: SKY, borderColor: '#0369A1', borderWidth: 0.8 },
        boxWidth: ['40%', '70%'],
      }],
    };
  }

  if (k === 'heatmap') {
    const series = spec.series ?? [];
    const xl = spec.x ?? [], yl = spec.y ?? [];
    const data = series.map((s: any) => [xl.indexOf(s.col), yl.indexOf(s.row), s.value]);
    const vals = series.map((s: any) => s.value);
    const vmin = vals.length ? Math.min(...vals, 0) : 0;
    const vmax = vals.length ? Math.max(...vals, 0) : 1;
    const sym = vmin < 0 && vmax > 0;
    const bound = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1;
    return {
      ...common,
      grid: { left: 1, right: 1, top: 1, bottom: 1, containLabel: false },
      xAxis: { type: 'category', data: xl, ...HIDDEN_AXIS },
      yAxis: { type: 'category', data: yl, ...HIDDEN_AXIS },
      visualMap: {
        show: false,
        min: sym ? -bound : vmin,
        max: sym ? bound : vmax,
        inRange: { color: sym ? ['#EF4444', '#ffffff', '#0D9488'] : ['#F5F5F4', '#0D9488'] },
      },
      series: [{ type: 'heatmap', data, itemStyle: { borderColor: '#fff', borderWidth: 0.5 } }],
    };
  }

  return null;
}

export default function SidebarChartThumbnail({ spec }: { spec: any }) {
  const option = thumbnailOption(spec);
  if (!option) return null;
  return (
    <ReactECharts
      option={option}
      style={{ width: '100%', height: 64 }}
      opts={{ renderer: 'svg' }}
    />
  );
}
