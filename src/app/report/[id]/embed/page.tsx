import EmbedClient from './EmbedClient';

export const metadata = {
  robots: { index: false, follow: false },
  other: { robots: 'noindex, indexifembedded' },
};

export default function EmbedPage({ params }: { params: { id: string } }) {
  return <EmbedClient id={params.id} />;
}
