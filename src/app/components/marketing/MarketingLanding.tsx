import MarketingNav from './MarketingNav';
import Hero from './Hero';
import TrustStrip from './TrustStrip';
import HowItWorks from './HowItWorks';
import ExampleReport from './ExampleReport';
import Features from './Features';
import UseCases from './UseCases';
import Pricing from './Pricing';
import Faq from './Faq';
import ClosingCta from './ClosingCta';
import MarketingFooter from './MarketingFooter';

export default function MarketingLanding() {
  return (
    <main className="min-h-screen bg-canvas text-ink">
      <MarketingNav />
      <Hero />
      <TrustStrip />
      <HowItWorks />
      <ExampleReport />
      <Features />
      <UseCases />
      <Pricing />
      <Faq />
      <ClosingCta />
      <MarketingFooter />
    </main>
  );
}
