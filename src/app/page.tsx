import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { getSupabaseServer } from './lib/supabase-server';
import MarketingLanding from './components/marketing/MarketingLanding';

export const metadata: Metadata = {
  title: 'ChartSage — Ten charts and a written summary, from one spreadsheet',
  description: 'Upload a CSV or Excel file. ChartSage picks the ten charts that matter and writes the analysis to match — no SQL, no pivot tables, no BI tool.',
  openGraph: {
    title: 'ChartSage — a report from your spreadsheet, in seconds',
    description: 'Upload a CSV. Get ten charts and a written summary. First report free.',
    type: 'website',
    images: ['/icon.svg'],
  },
  twitter: { card: 'summary_large_image' },
};

export default async function Home() {
  const supabase = getSupabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (user) redirect('/app');
  return <MarketingLanding />;
}
