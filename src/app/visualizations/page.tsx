'use client';

import { useEffect, useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';

const VisualizationCard = dynamic(() => import('./VisualizationCard'), { ssr: false });

const PROGRESS_STEPS = [
  { label: 'Uploading Data', key: 'upload' },
  { label: 'Getting AI Insights', key: 'ai' },
  { label: 'Creating Report', key: 'report' },
  { label: 'Done!', key: 'done' }
];

export default function VisualizationsPage() {
  const searchParams = useSearchParams();
  const [visualizations, setVisualizations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState(PROGRESS_STEPS[0].label);
  const plotRefs = useRef<(Plotly.PlotlyHTMLElement | null)[]>([]);

  // Helper to simulate progress
  const nextProgress = (step: number) => {
    setProgress(step);
    setProgressLabel(PROGRESS_STEPS[step].label);
  };

  // Main report generation effect
  useEffect(() => {
    // Load visualizations from query string
    try {
      const vizParam = searchParams.get('data');
      if (!vizParam) throw new Error('No report data found. Please generate a report first.');
      const decoded = JSON.parse(atob(decodeURIComponent(vizParam)));
      setVisualizations(decoded);
      setLoading(false);
    } catch (err: any) {
      setError(err.message || 'Failed to load report');
      setLoading(false);
    }
  }, [searchParams]);

  const createPlotlyConfig = (spec: any) => {
    return {
      displayModeBar: true,
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'] as any[]
    };
  };

  const createPlotlyLayout = (spec: any) => {
    const layout: any = {
      title: spec?.layout?.title || spec?.title || 'Untitled Chart',
      showlegend: spec?.style?.show_legend ?? true,
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      margin: { l: 50, r: 50, t: 50, b: 50 },
      width: undefined,  // Let it be responsive
      height: 400
    };

    if (spec?.layout?.xaxis_title) {
      layout.xaxis = { 
        title: spec.layout.xaxis_title,
        showgrid: spec?.layout?.show_grid ?? true
      };
    }
    if (spec?.layout?.yaxis_title) {
      layout.yaxis = { 
        title: spec.layout.yaxis_title,
        showgrid: spec?.layout?.show_grid ?? true
      };
    }
    if (spec?.layout?.barmode) {
      layout.barmode = spec.layout.barmode;
    }

    return layout;
  };

  const createPlotlyData = (spec: any) => {
    console.log('Creating plot data for spec:', spec);
    let data: any = {};

    if (!spec?.data) {
      console.error('Missing data in spec:', spec);
      return {
        type: 'scatter',
        x: [],
        y: [],
        name: 'Error: No data'
      };
    }

    // Get the actual data from the spec
    const xData = Array.isArray(spec.data.x) ? spec.data.x : 
                 spec.data.x ? [spec.data.x] : [];
    const yData = Array.isArray(spec.data.y) ? spec.data.y :
                 spec.data.y ? [spec.data.y] : [];

    switch (spec.type) {
      case 'bar':
        data = {
          type: 'bar',
          x: xData,
          y: yData,
          name: spec.title || 'Bar Chart',
          text: spec?.style?.show_values ? yData : undefined,
          textposition: 'auto',
          marker: {
            color: spec?.style?.color_scheme || 'rgb(55, 83, 109)'
          }
        };
        break;

      case 'line':
        data = {
          type: 'scatter',
          mode: spec?.style?.show_markers ? 'lines+markers' : 'lines',
          x: xData,
          y: yData,
          name: spec.title || 'Line Chart',
          line: {
            width: spec?.style?.line_width || 2,
            color: spec?.style?.color_scheme || 'rgb(55, 83, 109)'
          }
        };
        break;

      case 'scatter':
        data = {
          type: 'scatter',
          mode: 'markers',
          x: xData,
          y: yData,
          name: spec.title || 'Scatter Plot',
          marker: {
            size: spec?.style?.marker_size || 10,
            color: spec?.style?.color_scheme || 'rgb(55, 83, 109)'
          }
        };
        break;

      case 'pie':
        data = {
          type: 'pie',
          labels: spec.data.labels || xData,
          values: spec.data.values || yData,
          name: spec.title || 'Pie Chart',
          hole: spec?.style?.hole || 0,
          textinfo: spec?.style?.show_percentages ? 'label+percent' : 'label',
          marker: {
            colors: spec?.style?.color_scheme
          }
        };
        break;

      case 'box':
        console.log('Box plot yData:', yData);
        data = {
          type: 'box',
          y: yData,
          name: spec.title || 'Box Plot',
          boxpoints: spec?.style?.show_outliers ? 'outliers' : false,
          marker: {
            color: spec?.style?.color_scheme || 'rgb(55, 83, 109)'
          }
        };
        break;

      default:
        console.error('Unknown chart type:', spec.type);
        data = {
          type: 'scatter',
          x: [],
          y: [],
          name: 'Error: Unknown chart type'
        };
    }

    console.log('Generated plot data:', data);
    return data;
  };

  // Helper to download chart as PNG
  const handleDownload = async (index: number, title: string) => {
    const plotEl = plotRefs.current[index];
    if (plotEl) {
      // @ts-ignore
      const Plotly = await import('plotly.js-dist-min');
      Plotly.downloadImage(plotEl, {
        format: 'png',
        filename: title.replace(/\s+/g, '_').toLowerCase(),
        width: 900,
        height: 500,
        scale: 2
      });
    }
  };

  // Progress bar component
  const ProgressBar = ({ step }: { step: number }) => (
    <div className="w-full max-w-xl mx-auto mb-8">
      <div className="flex justify-between mb-2">
        {PROGRESS_STEPS.map((s, i) => (
          <div key={s.key} className={`text-xs font-semibold ${i <= step ? 'text-blue-700' : 'text-gray-400'}`}>{s.label}</div>
        ))}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div className="bg-blue-600 h-2 rounded-full transition-all duration-500" style={{ width: `${(step / (PROGRESS_STEPS.length - 1)) * 100}%` }}></div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mb-4"></div>
        <div className="text-center">
          <p className="text-xl font-semibold text-gray-900">Loading Report...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <div className="text-red-500 text-4xl mb-4">❌</div>
          <h2 className="text-xl font-semibold text-gray-900">Error Loading Report</h2>
          <p className="mt-2 text-gray-600">{error}</p>
          <button
            onClick={() => window.location.href = '/'}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Back to Upload
          </button>
        </div>
      </div>
    );
  }

  if (!visualizations?.length) {
    return <div className="flex justify-center items-center h-screen">No visualizations available</div>;
  }

  // --- Report Header ---
  const reportTitle = 'Data Insights Report';
  const reportSubtitle = 'Automatically generated visualizations for stakeholder review';
  const reportDate = new Date().toLocaleString();

  return (
    <div className="container mx-auto px-2 md:px-8 py-8 bg-gray-50 min-h-screen">
      {/* Report Header */}
      <header className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-2 tracking-tight">{reportTitle}</h1>
        <p className="text-lg md:text-xl text-gray-500 mb-1">{reportSubtitle}</p>
        <p className="text-sm text-gray-400">Generated: {reportDate}</p>
      </header>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {visualizations.map((viz, index) => {
          let plotData;
          if ((viz.type === 'bar' || viz.type === 'pie') && Array.isArray(viz.data)) {
            plotData = viz.data;
          } else {
            plotData = [createPlotlyData(viz)];
          }
          const layout = {
            ...createPlotlyLayout(viz),
            font: {
              family: 'Inter, sans-serif',
              size: 16,
              color: '#222'
            },
            legend: {
              font: { size: 15 },
              orientation: 'h',
              y: -0.2
            },
            margin: { l: 60, r: 40, t: 60, b: 60 },
            paper_bgcolor: '#fff',
            plot_bgcolor: '#fff',
            xaxis: {
              ...((createPlotlyLayout(viz).xaxis) || {}),
              tickfont: { size: 15 },
              titlefont: { size: 17 },
              showgrid: false,
              zeroline: false
            },
            yaxis: {
              ...((createPlotlyLayout(viz).yaxis) || {}),
              tickfont: { size: 15 },
              titlefont: { size: 17 },
              gridcolor: '#e5e7eb',
              zeroline: false
            },
            hoverlabel: { font: { size: 15 } },
            height: 400,
            width: undefined
          };
          const config = {
            ...createPlotlyConfig(viz),
            toImageButtonOptions: {
              format: 'png',
              filename: viz?.title?.replace(/\s+/g, '_').toLowerCase() || `chart_${index + 1}`,
              height: 500,
              width: 900,
              scale: 2
            }
          };
          return (
            <VisualizationCard
              key={index}
              viz={viz}
              index={index}
              plotData={plotData}
              handleDownload={handleDownload}
              layout={layout}
              config={config}
            />
          );
        })}
      </div>
      {/* Report Footer */}
      <footer className="mt-16 text-center text-gray-400 text-sm">
        <hr className="my-6 border-gray-200" />
        <p>Report generated by ChartSage &mdash; Data-driven insights for better decisions.</p>
      </footer>
    </div>
  );
} 