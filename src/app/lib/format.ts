export function formatShortCurrency(value: number): string {
  if (value == null || isNaN(value)) return '$0';
  const abs = Math.abs(value);
  if (abs >= 1e9) return '$' + (value / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
  return '$' + value.toFixed(2);
}

export function formatNumber(value: number): string {
  if (value == null || isNaN(value)) return '0';
  const abs = Math.abs(value);
  if (abs >= 1e9) return (value / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (value % 1 !== 0) {
    if (abs < 1) return value.toFixed(3).replace(/\.?0+$/, '');
    return value.toFixed(2).replace(/\.?0+$/, '');
  }
  return value.toLocaleString();
}

export function formatCount(value: number): string {
  if (value == null || isNaN(value)) return '0';
  return Math.round(value).toLocaleString();
}

export function formatPercentage(value: number): string {
  if (value == null || isNaN(value)) return '0%';
  // One decimal, but drop a trailing ".0" so ticks read "80%", not "80.0%".
  return (value * 100).toFixed(1).replace(/\.0$/, '') + '%';
}

export type YDisplayType = 'count' | 'currency' | 'percentage' | 'number';

export function getFormatter(t?: YDisplayType): (v: number) => string {
  switch (t) {
    case 'currency': return formatShortCurrency;
    case 'percentage': return formatPercentage;
    case 'count': return formatCount;
    case 'number':
    default: return formatNumber;
  }
}
