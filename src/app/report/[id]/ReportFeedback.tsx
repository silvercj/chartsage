'use client';

import { useState } from 'react';
import { posthog } from '../../lib/posthog';

export default function ReportFeedback({ reportId }: { reportId: string }) {
  const key = `cs_feedback_${reportId}`;
  const [done, setDone] = useState<boolean>(
    typeof window !== 'undefined' && sessionStorage.getItem(key) === '1',
  );
  const [rating, setRating] = useState<'up' | 'down' | null>(null);
  const [comment, setComment] = useState('');

  function send() {
    posthog.capture?.('report_feedback', { rating, comment: comment.trim() || undefined, reportId });
    try { sessionStorage.setItem(key, '1'); } catch {}
    setDone(true);
  }

  if (done) return <p className="text-sm text-ink-3 text-center py-4">Thanks for the feedback!</p>;

  return (
    <div className="card p-4 flex flex-col items-center gap-3 my-6">
      <span className="text-sm text-ink-2">Was this report useful?</span>
      <div className="flex gap-2">
        <button onClick={() => setRating('up')}
                className={`btn btn-ghost ${rating === 'up' ? 'border-accent text-accent' : ''}`}>👍</button>
        <button onClick={() => setRating('down')}
                className={`btn btn-ghost ${rating === 'down' ? 'border-accent text-accent' : ''}`}>👎</button>
      </div>
      {rating && (
        <div className="w-full max-w-md flex flex-col gap-2">
          <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={2}
                    placeholder="Anything we could improve? (optional)"
                    className="w-full bg-surface border border-line rounded-lg px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent outline-none transition-colors" />
          <button onClick={send} className="btn btn-primary self-end">Send</button>
        </div>
      )}
    </div>
  );
}
