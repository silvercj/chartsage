// PostHog → Evidence source: daily counts of key product events (last 90 days).
// Becomes the table `posthog.events_daily` (columns: day, event, events).

const HOST = process.env.EVIDENCE_POSTHOG_HOST;
const PROJECT = process.env.EVIDENCE_POSTHOG_PROJECT_ID;
const KEY = process.env.EVIDENCE_POSTHOG_API_KEY;

const query = `
  select
    toDate(timestamp) as day,
    event,
    count() as events
  from events
  where timestamp > now() - interval 90 day
    and event in (
      '$pageview',
      'report_generation_started',
      'report_generation_succeeded',
      'checkout_started',
      'buy_pack_clicked',
      'credits_spent'
    )
  group by day, event
  order by day, event
`;

const data = await queryPostHog(query);

export { data };

async function queryPostHog(hogql) {
  if (!HOST || !PROJECT || !KEY) {
    throw new Error(
      'PostHog env vars missing — set EVIDENCE_POSTHOG_HOST, EVIDENCE_POSTHOG_PROJECT_ID and EVIDENCE_POSTHOG_API_KEY in bi/.env'
    );
  }
  // Plain concatenation (not template literals): Evidence preprocesses source files
  // for dollar-brace interpolations, so backtick interpolation spams warnings / risks mangling.
  const res = await fetch(HOST + '/api/projects/' + PROJECT + '/query/', {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: { kind: 'HogQLQuery', query: hogql } })
  });
  if (!res.ok) {
    throw new Error('PostHog query failed (' + res.status + '): ' + (await res.text()));
  }
  const json = await res.json();
  const cols = json.columns ?? [];
  const rows = (json.results ?? []).map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i]])));
  return rows.length ? rows : [Object.fromEntries(cols.map((c) => [c, null]))];
}
