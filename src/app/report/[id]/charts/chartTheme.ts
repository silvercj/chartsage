// Shared ECharts styling for the light report canvas. Charts keep their own
// data/series logic; this centralizes palette, axes, tooltip, and fonts.

export const CHART_PALETTE = ['#0C5C52', '#5B8C7E', '#C99A3F', '#B5673A', '#3E5C6B', '#9DB7AE'];
export const CHART_TEAL = '#0C5C52';
export const CHART_OCHRE = '#C99A3F';
export const CHART_INK = '#1B1A16';
export const CHART_INK_MUTED = '#9A9183';
export const CHART_LINE = '#E6E0D4';

// ECharts canvas needs a literal font-family string; resolve the Geist Mono
// family that next/font set on --font-geist-mono (falls back gracefully on SSR).
export function monoFamily(): string {
  if (typeof window === 'undefined') return 'ui-monospace, monospace';
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue('--font-geist-mono')
    .trim();
  return v ? `${v}, ui-monospace, monospace` : 'ui-monospace, monospace';
}

// Base option fragment every chart spreads in.
export function chartBase() {
  const mono = monoFamily();
  return {
    color: CHART_PALETTE,
    textStyle: { fontFamily: mono, color: CHART_INK_MUTED },
    grid: { left: 8, right: 18, top: 20, bottom: 8, containLabel: true },
    tooltip: {
      backgroundColor: '#ffffff',
      borderColor: CHART_LINE,
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { color: CHART_INK, fontFamily: mono, fontSize: 12 },
      extraCssText: 'box-shadow:0 8px 24px -8px rgba(27,26,22,.22);border-radius:10px;',
    },
  };
}

// Category axis with stripped chrome + mono labels. `interval`/`rotate` for density.
export function catAxis(extra: Record<string, any> = {}) {
  const { axisLabel, ...rest } = extra;
  return {
    type: 'category',
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: { fontFamily: monoFamily(), fontSize: 11, color: CHART_INK_MUTED, ...(axisLabel || {}) },
    ...rest,
  };
}

// How far to slant category-axis labels so they don't collide. Weighs both the
// count AND the length — long names ("Netherlands") overlap well before you have 10
// of them. Returns 0 (flat) when they comfortably fit. Pair with grid containLabel.
export function labelRotation(cats: any[]): number {
  const n = cats?.length ?? 0;
  if (n === 0) return 0;
  const maxLen = cats.reduce((mx: number, s: any) => Math.max(mx, String(s ?? '').length), 0);
  const crowded = n > 10 || n * maxLen > 55;
  return crowded ? (n > 16 || maxLen > 16 ? 45 : 30) : 0;
}

// Value axis with hairline split-lines + mono labels.
export function valAxis(extra: Record<string, any> = {}) {
  const { axisLabel, ...rest } = extra;
  return {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { lineStyle: { color: CHART_LINE, width: 1 } },
    axisLabel: { fontFamily: monoFamily(), fontSize: 11, color: CHART_INK_MUTED, ...(axisLabel || {}) },
    ...rest,
  };
}

// Teal vertical area gradient (object form — no echarts import needed).
export function tealAreaGradient() {
  return {
    type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
    colorStops: [
      { offset: 0, color: 'rgba(12,92,82,0.22)' },
      { offset: 1, color: 'rgba(12,92,82,0)' },
    ],
  };
}
