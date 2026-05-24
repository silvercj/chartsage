interface Props {
  summary: string;
  generatedAt: string;
}

export default function ReportSummary({ summary, generatedAt }: Props) {
  const paragraphs = summary.split(/\n\s*\n/).filter((p) => p.trim());
  const date = new Date(generatedAt).toLocaleString();
  return (
    <header className="mb-8">
      <div className="text-center mb-6">
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">Data Report</h1>
        <p className="text-sm text-gray-400">Generated: {date}</p>
      </div>
      <div className="max-w-3xl mx-auto space-y-4 text-gray-700 leading-relaxed">
        {paragraphs.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>
    </header>
  );
}
