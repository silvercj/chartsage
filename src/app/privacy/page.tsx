import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Privacy Policy · ChartSage' };

export default function PrivacyPage() {
  // NOT LEGAL ADVICE — starting-point template; have a professional review before the paid launch.
  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="max-w-2xl mx-auto px-6 py-16">
        <h1 className="font-display text-3xl font-semibold mb-2">Privacy Policy</h1>
        <p className="font-mono text-xs text-ink-3 mb-10">Last updated: June 2, 2026</p>
        <div className="space-y-6 text-ink-2 leading-relaxed text-sm">
          <section><h2 className="font-display text-lg text-ink mb-2">1. What we collect</h2>
            <p>We collect your account email; the files you upload (CSV/Excel); the reports we generate for you; your IP address and a coarse device fingerprint (used to prevent abuse of the free tier); and product usage analytics.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">2. How we use it</h2>
            <p>We use this information to provide and operate the Service, including processing your data through a third-party AI provider to generate your reports.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">3. Data handling &amp; retention</h2>
            <p>Your data is stored in our cloud infrastructure and retained while your account is active and as needed to operate the Service. Uploaded files and generated reports are kept so you can revisit your reports.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">4. Deletion</h2>
            <p>You can request deletion of your account and associated data by emailing us at <span className="font-mono">support@chartsage.app</span> (update with the real address). Self-serve deletion is coming.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">5. Cookies &amp; analytics</h2>
            <p>We use cookies and a product-analytics tool to understand usage and improve the Service.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">6. Changes</h2>
            <p>We may update this Policy; material changes will be reflected by the &ldquo;last updated&rdquo; date above.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">7. Contact</h2>
            <p>Questions about your privacy? Contact us at <span className="font-mono">support@chartsage.app</span> (update with your real support address).</p></section>
        </div>
      </div>
    </main>
  );
}
