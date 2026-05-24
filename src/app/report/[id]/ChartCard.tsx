'use client';

import dynamic from 'next/dynamic';

const BarChart = dynamic(() => import('./charts/BarChart'), { ssr: false });
const HistogramChart = dynamic(() => import('./charts/HistogramChart'), { ssr: false });
const ScatterChart = dynamic(() => import('./charts/ScatterChart'), { ssr: false });
const LineChart = dynamic(() => import('./charts/LineChart'), { ssr: false });
const PieChart = dynamic(() => import('./charts/PieChart'), { ssr: false });
const BoxPlot = dynamic(() => import('./charts/BoxPlot'), { ssr: false });
const Heatmap = dynamic(() => import('./charts/Heatmap'), { ssr: false });

interface Props {
  spec: any;
  caption: string;
}

export default function ChartCard({ spec, caption }: Props) {
  const renderer = (() => {
    switch (spec.kind) {
      case 'bar': return <BarChart spec={spec} />;
      case 'histogram': return <HistogramChart spec={spec} />;
      case 'scatter': return <ScatterChart spec={spec} />;
      case 'line': return <LineChart spec={spec} />;
      case 'pie': return <PieChart spec={spec} />;
      case 'box': return <BoxPlot spec={spec} />;
      case 'heatmap': return <Heatmap spec={spec} />;
      default: return <p className="text-sm text-red-600">Unsupported chart kind: {String(spec.kind)}</p>;
    }
  })();

  return (
    <section className="bg-white rounded-2xl shadow-md border border-gray-200 p-6 flex flex-col">
      <h2 className="text-lg font-bold text-gray-900 mb-2">{spec.title}</h2>
      <div className="flex-1 min-h-[300px]">{renderer}</div>
      <p className="text-sm text-gray-600 mt-3 italic">{caption}</p>
    </section>
  );
}
