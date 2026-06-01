'use client';
import dynamic from 'next/dynamic';

const BarChart = dynamic(() => import('./BarChart'), { ssr: false });
const HistogramChart = dynamic(() => import('./HistogramChart'), { ssr: false });
const ScatterChart = dynamic(() => import('./ScatterChart'), { ssr: false });
const LineChart = dynamic(() => import('./LineChart'), { ssr: false });
const PieChart = dynamic(() => import('./PieChart'), { ssr: false });
const BoxPlot = dynamic(() => import('./BoxPlot'), { ssr: false });
const Heatmap = dynamic(() => import('./Heatmap'), { ssr: false });

export default function ChartContent({ spec }: { spec: any }) {
  switch (spec.kind) {
    case 'bar': return <BarChart spec={spec} />;
    case 'histogram': return <HistogramChart spec={spec} />;
    case 'scatter': return <ScatterChart spec={spec} />;
    case 'line': return <LineChart spec={spec} />;
    case 'pie': return <PieChart spec={spec} />;
    case 'box': return <BoxPlot spec={spec} />;
    case 'heatmap': return <Heatmap spec={spec} />;
    default: return <p className="text-sm text-ember">Unsupported chart kind: {String(spec.kind)}</p>;
  }
}
