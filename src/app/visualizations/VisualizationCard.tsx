import Plot from 'react-plotly.js';
import { useRef } from 'react';

interface VisualizationCardProps {
  viz: any;
  index: number;
  plotData: any;
  handleDownload: (index: number, title: string) => void;
  layout: any;
  config: any;
}

export default function VisualizationCard({ viz, index, plotData, handleDownload, layout, config }: VisualizationCardProps) {
  const plotRef = useRef<any>(null);

  return (
    <section
      className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 flex flex-col justify-between min-h-[480px]"
      aria-label={viz?.title || `Chart ${index + 1}`}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-1 leading-tight">{viz?.title || `Chart ${index + 1}`}</h2>
          <p className="text-md text-blue-700 mb-2 font-medium">{viz?.description || 'No description available'}</p>
        </div>
        <button
          className="ml-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg shadow transition"
          onClick={() => handleDownload(index, viz?.title || `chart_${index + 1}`)}
          aria-label={`Download ${viz?.title || `Chart ${index + 1}`} as PNG`}
        >
          Download
        </button>
      </div>
      <div className="flex-1 flex items-center justify-center min-h-[320px]">
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          useResizeHandler={true}
          className="w-full"
          ref={plotRef}
        />
      </div>
    </section>
  );
} 