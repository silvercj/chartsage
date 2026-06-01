import MarketingNav from './MarketingNav';
import Hero from './Hero';
import TrustStrip from './TrustStrip';

export default function MarketingLanding() {
  return (
    <main className="min-h-screen bg-canvas text-ink">
      <MarketingNav />
      <Hero />
      <TrustStrip />
    </main>
  );
}
