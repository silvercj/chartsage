'use client';
import { useState } from 'react';

interface Props {
  summary: string;
  generatedAt: string;
}

export default function ReportSummary({ summary, generatedAt }: Props) {
  const [open, setOpen] = useState(true);
  const paragraphs = summary.split(/\n\s*\n/).filter((p) => p.trim());
  const date = new Date(generatedAt).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <header className="mb-10">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-baseline justify-between pb-6 border-b border-line group cursor-pointer"
        aria-expanded={open}
      >
        <div className="flex items-baseline gap-3">
          <svg
            className={`w-4 h-4 text-ink-3 transition-transform ${open ? 'rotate-90' : ''}`}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <h1 className="font-display text-3xl font-medium text-ink">
            Insights
          </h1>
        </div>
        <span className="font-mono text-xs uppercase tracking-wide text-ink-3">{date}</span>
      </button>
      <div
        className={`overflow-hidden transition-all duration-300 ${open ? 'max-h-[2000px] opacity-100 mt-6' : 'max-h-0 opacity-0'}`}
      >
        <div className="max-w-3xl space-y-4 text-ink-2 leading-relaxed text-[15px]">
          {paragraphs.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      </div>
    </header>
  );
}
