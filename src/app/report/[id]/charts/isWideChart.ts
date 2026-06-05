// A single-series bar chart with many categories renders as a full-width
// horizontal ranking (see 2026-06-05-collapsible-wide-charts-design). The card
// uses this to decide full-width layout + show the collapse/expand toggle.
export function isWideChart(spec: any): boolean {
  return spec?.kind === 'bar' && (spec?.x?.length ?? 0) > 12;
}
