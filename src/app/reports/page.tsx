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
      <div className="min-h-screen bg-stone-50 flex items-center justify-center">
        <p className="text-stone-600">{error}</p>
      </div>
    );
  }
  if (!reports) {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-stone-300 border-t-stone-900" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-16">
        <header className="mb-8 flex items-baseline justify-between">
          <h1 className="text-3xl font-semibold tracking-tight text-stone-900">My reports</h1>
          <a href="/" className="text-sm text-stone-500 hover:text-stone-900">New report →</a>
        </header>

        {reports.length === 0 ? (
          <div className="p-8 bg-white border border-stone-200 rounded-2xl text-center">
            <p className="text-stone-600 mb-4">You haven't created any reports yet.</p>
            <a href="/" className="inline-block px-5 py-2.5 bg-stone-900 text-white text-sm rounded-lg hover:bg-stone-800">
              Create your first report
            </a>
          </div>
        ) : (
          <ul className="space-y-3">
            {reports.map((r) => (
              <li key={r.id}>
                <a
                  href={`/report/${r.id}`}
                  className="block p-5 bg-white border border-stone-200 rounded-2xl hover:border-stone-300 hover:shadow-sm transition-all"
                >
                  <div className="flex items-baseline justify-between gap-4">
                    <p className="font-medium text-stone-900 truncate">{r.title}</p>
                    <span className="text-xs text-stone-400 flex-shrink-0">
                      {r.createdAt ? new Date(r.createdAt).toLocaleDateString() : ''}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2 flex-wrap">
                    <span className="text-xs text-stone-500">{r.chartCount} charts</span>
                    {r.kinds.slice(0, 5).map((k) => (
                      <span key={k} className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-stone-100 text-stone-500">
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
