// PostHog → Evidence source: event volume over the last 90 days.
// Evidence runs this file at `npm run sources` (top-level await is supported);
// the named `data` export (an array of objects) becomes the table `posthog.event_totals`.
// Credentials come from bi/.env (EVIDENCE_-prefixed vars are injected by Evidence).

const HOST = process.env.EVIDENCE_POSTHOG_HOST;
const PROJECT = process.env.EVIDENCE_POSTHOG_PROJECT_ID;
const KEY = process.env.EVIDENCE_POSTHOG_API_KEY;

const query = `
  select
    event,
    count() as events,
    count(distinct person_id) as users
  from events
  where timestamp > now() - interval 90 day
  group by event
  order by events desc
  limit 100
`;

const data = await queryPostHog(query);

export { data };

// --- shared helper (inlined: every .js in a source dir must itself export `data`,
//     so a separate helper module isn't possible here) ----------------------------
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
  // Evidence's JS connector reads Object.keys(data[0]), so never hand it an empty array.
  return rows.length ? rows : [Object.fromEntries(cols.map((c) => [c, null]))];
}
