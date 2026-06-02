'use client';
import { useRouter } from 'next/navigation';
import { posthog } from '../lib/posthog';

export default function OutOfCreditsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  if (!open) return null;

  function buy() {
    posthog.capture?.('buy_credits_clicked', { source: 'out_of_credits_modal' });
    onClose();
    router.push('/credits');
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4" onClick={onClose}>
      <div className="card shadow-card-lg rounded-2xl p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h2 className="font-display text-xl text-ember mb-1">You're out of credits</h2>
        <p className="text-sm text-ink-2 mb-5">
          Top up to keep generating reports — credit packs start at £5.
        </p>
        <button type="button" onClick={buy} className="btn btn-primary w-full">
          Buy credits
        </button>
        <button type="button" onClick={onClose} className="mt-4 w-full text-sm text-ink-2 hover:text-ink">
          Maybe later
        </button>
      </div>
    </div>
  );
}
