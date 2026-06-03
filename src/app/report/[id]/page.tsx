import type { Metadata } from 'next';
import ReportClient from './ReportClient';

const API = process.env.NEXT_PUBLIC_API_URL!;
const SITE = process.env.NEXT_PUBLIC_SITE_URL || 'https://chartsage.app';

export async function generateMetadata({ params }: { params: { id: string } }): Promise<Metadata> {
  try {
    const res = await fetch(`${API}/report/${params.id}/meta`, { cache: 'no-store' });
    if (!res.ok) return { robots: { index: false, follow: false } };
    const m: { is_public?: boolean; title?: string; description?: string; og_image_url?: string | null } = await res.json();
    if (!m.is_public) return { robots: { index: false, follow: false } };
    const url = `${SITE}/report/${params.id}`;
    const images = m.og_image_url ? [{ url: m.og_image_url, width: 1200, height: 630 }] : undefined;
    return {
      title: m.title,
      description: m.description,
      alternates: { canonical: url },
      robots: { index: true, follow: true },
      openGraph: { title: m.title, description: m.description, url, type: 'article', images },
      twitter: { card: 'summary_large_image', title: m.title, description: m.description, images: m.og_image_url ? [m.og_image_url] : undefined },
    };
  } catch {
    return { robots: { index: false, follow: false } };
  }
}

export default function Page({ params }: { params: { id: string } }) {
  return <ReportClient params={params} />;
}
