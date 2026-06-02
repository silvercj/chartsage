import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Terms of Service · ChartSage' };

export default function TermsPage() {
  // NOT LEGAL ADVICE — starting-point template; have a professional review before the paid launch.
  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="max-w-2xl mx-auto px-6 py-16">
        <h1 className="font-display text-3xl font-semibold mb-2">Terms of Service</h1>
        <p className="font-mono text-xs text-ink-3 mb-10">Last updated: June 2, 2026</p>
        <div className="space-y-6 text-ink-2 leading-relaxed text-sm">
          <section><h2 className="font-display text-lg text-ink mb-2">1. Acceptance</h2>
            <p>By accessing or using ChartSage (&ldquo;the Service&rdquo;) you agree to these Terms. If you do not agree, do not use the Service.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">2. The Service</h2>
            <p>ChartSage generates data visualizations and written analysis from files you upload, using automated AI processing. Output is generated automatically and may contain errors — review it before relying on it for any decision.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">3. Accounts &amp; credits</h2>
            <p>Some features require an account and consume credits. Credits have no cash value, are non-transferable, and during the beta are provided at our discretion.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">4. Acceptable use</h2>
            <p>Do not abuse, overload, or attempt to circumvent the usage limits of the Service (including automated or large-scale generation of free reports), upload unlawful content, or infringe others&rsquo; rights.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">5. Your content</h2>
            <p>You retain ownership of the files you upload and the reports generated for you. You grant us the limited rights needed to operate the Service — storing and processing your data to produce your reports. See our <a className="text-accent underline" href="/privacy">Privacy Policy</a>.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">6. Disclaimers &amp; liability</h2>
            <p>The Service and its AI-generated output are provided &ldquo;as is&rdquo;, without warranties of any kind. To the maximum extent permitted by law, we are not liable for any indirect or consequential damages arising from your use of the Service or reliance on its output.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">7. Changes</h2>
            <p>We may update these Terms; material changes will be reflected by the &ldquo;last updated&rdquo; date above.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">8. Contact</h2>
            <p>Questions about these Terms? Contact us at <span className="font-mono">support@chartsage.app</span> (update with your real support address).</p></section>
        </div>
      </div>
    </main>
  );
}
