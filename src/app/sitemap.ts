import type { MetadataRoute } from 'next';

const SITE = process.env.NEXT_PUBLIC_SITE_URL || 'https://chartsage.app';
const API = process.env.NEXT_PUBLIC_API_URL;

export const revalidate = 3600;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const paths = ['', '/app', '/terms', '/privacy', '/contact', '/credits'];
  const staticRoutes: MetadataRoute.Sitemap = paths.map((p) => ({
    url: `${SITE}${p}`,
    changeFrequency: 'weekly' as const,
    priority: p === '' ? 1 : 0.6,
  }));

  let reports: MetadataRoute.Sitemap = [];
  try {
    if (API) {
      const res = await fetch(`${API}/reports/public`, { next: { revalidate: 3600 } });
      if (res.ok) {
        const rows: { id: string }[] = await res.json();
        reports = rows.map((r) => ({
          url: `${SITE}/report/${r.id}`,
          changeFrequency: 'monthly' as const,
          priority: 0.5,
        }));
      }
    }
  } catch {
    // sitemap still serves the static routes
  }
  return [...staticRoutes, ...reports];
}
