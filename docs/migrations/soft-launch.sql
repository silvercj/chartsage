-- Soft-launch: anonymous report log (abuse tracking + daily caps). Run once in Supabase SQL editor.
create table if not exists anon_report_log (
  id          uuid primary key default gen_random_uuid(),
  anon_id     uuid,
  ip          text,
  fingerprint text,
  created_at  timestamptz not null default now()
);
create index if not exists anon_report_log_created_idx on anon_report_log (created_at desc);
create index if not exists anon_report_log_ip_idx on anon_report_log (ip, created_at desc);
alter table anon_report_log enable row level security;  -- service-role only; no client policies
