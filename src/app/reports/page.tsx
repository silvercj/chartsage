'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch } from '../lib/api';
import { useCredits } from '../lib/useCredits';

interface ReportRow {
  id: string;
  title: string;
  chartCount: number;
  kinds: string[];
  createdAt: string | null;
}

export default function MyReportsPage() {
  const router = useRouter();
  const { session, authLoading } = useCredits();
  const [reports, setReports] = useState<ReportRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (authLoading) return;                 // wait until auth state is known
    if (!session) { router.replace('/login?next=/reports'); return; }
    (async () => {
      try {
        const res = await apiFetch('/my-reports');
        if (res.status === 401) {
          router.replace('/login?next=/reports');
          return;
        }
        if (!res.ok) {
          setError('Could not load your reports.');
          return;
        }
        setReports(await res.json());
      } catch {
        setError('Could not load your reports.');
      }
    })();
  }, [authLoading, session, router]);

  if (error) {
    return (
      <div className="min-h-screen bg-canvas flex items-center justify-center">
        <p className="text-ink-3">{error}</p>
      </div>
    );
  }
  if (!reports) {
    return (
      <div className="min-h-screen bg-canvas flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-line-2 border-t-accent" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-canvas">
      <div className="max-w-5xl mx-auto px-6 py-12">
        <header className="mb-8 flex items-baseline justify-between">
          <h1 className="font-display text-3xl font-medium text-ink">My reports</h1>
          <a href="/app" className="text-sm text-ink-2 hover:text-ink transition-colors">New report →</a>
        </header>

        {reports.length === 0 ? (
          <div className="card shadow-card rounded-2xl p-8 text-center">
            <p className="text-ink-3 mb-4">You haven't created any reports yet.</p>
            <a href="/app" className="btn btn-primary">
              Create your first report
            </a>
          </div>
        ) : (
          <ul className="space-y-3">
            {reports.map((r) => (
              <li key={r.id}>
                <a
                  href={`/report/${r.id}`}
                  className="card shadow-card hover:border-line-2 transition-colors rounded-2xl p-5 block"
                >
                  <div className="flex items-baseline justify-between gap-4">
                    <p className="font-display text-lg text-ink truncate">{r.title}</p>
                    <span className="font-mono text-xs text-ink-3 flex-shrink-0">
                      {r.createdAt ? new Date(r.createdAt).toLocaleDateString() : ''}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-xs text-ink-3">{r.chartCount} charts</span>
                    {r.kinds.slice(0, 5).map((k) => (
                      <span key={k} className="font-mono text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-surface-2 text-ink-3">
                        {k}
                      </span>
                    ))}
                  </div>
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
