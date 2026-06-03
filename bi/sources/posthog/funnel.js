// PostHog → Evidence source: activation funnel (unique users per step, last 90 days).
// Becomes the table `posthog.funnel` (columns: step_order, step, users).
// Steps are fixed and ordered here so the chart always renders in funnel order,
// even if a step has zero users.
//
// NB: plain string concatenation (not template literals) — Evidence preprocesses
// source files for dollar-brace interpolations, so backtick interpolation spams warnings.

const HOST = process.env.EVIDENCE_POSTHOG_HOST;
const PROJECT = process.env.EVIDENCE_POSTHOG_PROJECT_ID;
const KEY = process.env.EVIDENCE_POSTHOG_API_KEY;

const STEPS = [
  ['$pageview', 'Visited'],
  ['report_generation_started', 'Started a report'],
  ['report_generation_succeeded', 'Got a report'],
  ['checkout_started', 'Started checkout']
];

const inList = STEPS.map(function (s) {
  return "'" + s[0] + "'";
}).join(', ');

const query =
  'select event, count(distinct person_id) as users ' +
  'from events ' +
  'where timestamp > now() - interval 90 day ' +
  'and event in (' + inList + ') ' +
  'group by event';

const rows = await queryPostHog(query);
const usersByEvent = Object.fromEntries(rows.map((r) => [r.event, Number(r.users) || 0]));

const data = STEPS.map(function (entry, i) {
  const event = entry[0];
  const label = entry[1];
  return {
    step_order: i + 1,
    step: i + 1 + '. ' + label,
    users: usersByEvent[event] ?? 0
  };
});

export { data };

async function queryPostHog(hogql) {
  if (!HOST || !PROJECT || !KEY) {
    throw new Error(
      'PostHog env vars missing — set EVIDENCE_POSTHOG_HOST, EVIDENCE_POSTHOG_PROJECT_ID and EVIDENCE_POSTHOG_API_KEY in bi/.env'
    );
  }
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
  return (json.results ?? []).map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i]])));
}
