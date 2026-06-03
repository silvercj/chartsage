import OgCard from './OgCard';

// Internal capture target for the social-preview image (screenshotted server-side by
// Playwright at 1200x630). Never indexed; not linked anywhere public.
export const metadata = {
  robots: { index: false, follow: false },
};

export default function OgPage({ params }: { params: { id: string } }) {
  return <OgCard id={params.id} />;
}
