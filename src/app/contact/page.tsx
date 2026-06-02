'use client';

import { useState } from 'react';
import { apiFetch } from '../lib/api';
import { posthog } from '../lib/posthog';

export default function ContactPage() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [company, setCompany] = useState('');   // honeypot
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!message.trim()) { setStatus('error'); return; }
    setStatus('sending');
    try {
      const res = await apiFetch('/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, message, company }),
      });
      if (!res.ok) { setStatus('error'); return; }
      posthog.capture?.('contact_submitted', {});
      setStatus('sent');
    } catch {
      setStatus('error');
    }
  }

  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="max-w-lg mx-auto px-6 py-16">
        <h1 className="font-display text-3xl font-semibold mb-2">Contact us</h1>
        <p className="text-sm text-ink-2 mb-8">Questions, bugs, or a deletion request? Send us a message and we&rsquo;ll get back to you.</p>
        {status === 'sent' ? (
          <div className="card p-6"><p className="text-ink">Thanks &mdash; we&rsquo;ll get back to you.</p></div>
        ) : (
          <form onSubmit={submit} className="space-y-4">
            <label className="block">
              <span className="block text-xs text-ink-3 mb-1">Your email (so we can reply)</span>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com"
                     className="w-full bg-surface-2 border border-line-2 rounded-lg px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent outline-none transition-colors" />
            </label>
            <label className="block">
              <span className="block text-xs text-ink-3 mb-1">Message</span>
              <textarea value={message} onChange={(e) => setMessage(e.target.value)} required rows={6} placeholder="How can we help?"
                        className="w-full bg-surface-2 border border-line-2 rounded-lg px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent outline-none transition-colors" />
            </label>
            <input type="text" name="company" value={company} onChange={(e) => setCompany(e.target.value)}
                   tabIndex={-1} autoComplete="off" aria-hidden="true"
                   className="absolute -left-[9999px] h-0 w-0 opacity-0" />
            {status === 'error' && <p className="text-sm text-ember">Couldn&rsquo;t send &mdash; please add a message and try again.</p>}
            <button type="submit" disabled={status === 'sending'} className="btn btn-primary">
              {status === 'sending' ? 'Sending…' : 'Send message'}
            </button>
          </form>
        )}
      </div>
    </main>
  );
}
