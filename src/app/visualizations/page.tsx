'use client';

import { useEffect, useState } from 'react';
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

  // Helper to simulate progress
  const nextProgress = (step: number) => {
    setProgress(step);
    setProgressLabel(PROGRESS_STEPS[step].label);
  };

  // Main report generation effect
  useEffect(() => {
    // Load visualizations from session ID in query string
    const sessionId = searchParams.get('session');
    if (!sessionId) {
      setError('No report session found. Please generate a report first.');
      setLoading(false);
      return;
    }
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/session-dashboard/${sessionId}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to load dashboard session');
        return res.json();
      })
      .then(data => {
        setVisualizations(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || 'Failed to load report');
        setLoading(false);
      });
  }, [searchParams]);

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
      <div id="dashboard-root">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {visualizations.map((viz, index) => {
            let plotData;
            if (viz.type === 'pie' && typeof viz.data === 'object' && !Array.isArray(viz.data)) {
              plotData = viz.data;
            } else if (viz.type === 'bar' && Array.isArray(viz.data)) {
              // For grouped bar charts, you may want to handle multiple traces
              // For now, use the first trace
              plotData = viz.data[0];
            } else if (typeof viz.data === 'object') {
              plotData = viz.data;
            } else {
              plotData = {};
            }
            const layout = {
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
                tickfont: { size: 15 },
                titlefont: { size: 17 },
                showgrid: false,
                zeroline: false
              },
              yaxis: {
                tickfont: { size: 15 },
                titlefont: { size: 17 },
                gridcolor: '#e5e7eb',
                zeroline: false
              },
              hoverlabel: { font: { size: 15 } },
              height: 400,
              width: undefined
            };
            return (
              <VisualizationCard
                key={index}
                viz={viz}
                index={index}
                plotData={plotData}
                layout={layout}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
} 