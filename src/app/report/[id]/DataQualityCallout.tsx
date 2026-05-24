interface Props {
  notes: string[];
}

export default function DataQualityCallout({ notes }: Props) {
  return (
    <aside className="max-w-3xl mx-auto bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-6">
      <h3 className="font-semibold text-yellow-900 mb-2">Data quality notes</h3>
      <ul className="list-disc pl-5 space-y-1 text-yellow-900 text-sm">
        {notes.map((n, i) => (
          <li key={i}>{n}</li>
        ))}
      </ul>
    </aside>
  );
}
