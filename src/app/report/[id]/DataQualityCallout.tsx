interface Props {
  notes: string[];
}

export default function DataQualityCallout({ notes }: Props) {
  return (
    <aside className="max-w-3xl bg-surface-2 border border-line rounded-xl p-4 mt-2">
      <div className="flex items-start gap-3">
        <svg
          className="w-5 h-5 text-ember flex-shrink-0 mt-0.5"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M5.07 19h13.86c1.54 0 2.5-1.67 1.73-3L13.73 4a2 2 0 00-3.46 0L3.34 16c-.77 1.33.19 3 1.73 3z" />
        </svg>
        <div className="flex-1">
          <h3 className="font-mono text-xs uppercase tracking-wide text-ink-3 mb-1.5">
            Data quality
          </h3>
          <ul className="space-y-1.5 text-sm text-ink-2 leading-relaxed">
            {notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      </div>
    </aside>
  );
}
