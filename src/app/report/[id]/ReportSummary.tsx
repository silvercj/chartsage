interface Props {
  summary: string;
  generatedAt: string;
}

export default function ReportSummary({ summary, generatedAt }: Props) {
  const paragraphs = summary.split(/\n\s*\n/).filter((p) => p.trim());
  const date = new Date(generatedAt).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <header className="mb-10">
      <div className="flex items-baseline justify-between mb-6 pb-6 border-b border-stone-200">
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-stone-900">
          Insights
        </h1>
        <span className="text-xs uppercase tracking-widest text-stone-400">
          {date}
        </span>
      </div>
      <div className="max-w-3xl space-y-4 text-stone-700 leading-relaxed text-[15px]">
        {paragraphs.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>
    </header>
  );
}
