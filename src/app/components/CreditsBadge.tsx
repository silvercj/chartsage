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
      className={`pill ${low ? '!border-ember/50 !text-ember' : ''}`}
    >
      <span aria-hidden className={low ? '' : 'text-ember'}>⚡</span>{balance}
    </a>
  );
}
