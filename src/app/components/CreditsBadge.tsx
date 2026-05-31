'use client';
import { useCredits } from '../lib/useCredits';
import { REPORT_COST } from '../lib/credits';

export default function CreditsBadge() {
  const { balance } = useCredits();
  if (balance === null) return null;
  const low = balance < REPORT_COST;
  return (
    <a
      href="/credits"
      title="Your credit balance"
      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-sm font-medium ring-1 ${
        low ? 'bg-amber-50 text-amber-700 ring-amber-200' : 'bg-stone-100 text-stone-700 ring-stone-200'
      }`}
    >
      <span aria-hidden>⚡</span>{balance}
    </a>
  );
}
