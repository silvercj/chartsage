import MarketingNav from './MarketingNav';
import Hero from './Hero';
import TrustStrip from './TrustStrip';
import HowItWorks from './HowItWorks';
import Features from './Features';
import UseCases from './UseCases';

export default function MarketingLanding() {
  return (
    <main className="min-h-screen bg-canvas text-ink">
      <MarketingNav />
      <Hero />
      <TrustStrip />
      <HowItWorks />
      <Features />
      <UseCases />
    </main>
  );
}
