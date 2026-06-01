import { redirect } from 'next/navigation';
import { getSupabaseServer } from './lib/supabase-server';
import MarketingLanding from './components/marketing/MarketingLanding';

export default async function Home() {
  const supabase = getSupabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (user) redirect('/app');
  return <MarketingLanding />;
}
