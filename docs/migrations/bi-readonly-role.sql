-- docs/migrations/bi-readonly-role.sql
-- Local BI (Evidence): a READ-ONLY Postgres role for ChartSage's Supabase DB.
-- Run once in Supabase → SQL editor. Lets the BI tool read every row of the
-- analytics tables while being unable to write anything.
--
-- Why the policies? Most ChartSage tables have RLS enabled (see sp3-credits.sql,
-- soft-launch.sql, support.sql, payments.sql). A plain read-only role is allowed
-- to SELECT, but RLS would still return ZERO rows. The permissive policies below
-- grant THIS role full read access; they are scoped to `chartsage_readonly`, so
-- behaviour for anon/auth app users is unchanged. Safe to re-run.

-- 1) The role. ⚠️ CHANGE THE PASSWORD before running.
do $$
begin
  if not exists (select from pg_roles where rolname = 'chartsage_readonly') then
    create role chartsage_readonly login password 'CHANGE_ME_to_a_strong_password';
  end if;
end $$;

-- 2) Allow it to connect and read the public schema.
grant connect on database postgres to chartsage_readonly;
grant usage  on schema public      to chartsage_readonly;
grant select on all tables in schema public to chartsage_readonly;
alter default privileges in schema public grant select on tables to chartsage_readonly;

-- 3) Permissive read policies so RLS doesn't hide rows from this role.
do $$
declare t text;
begin
  foreach t in array array[
    'profiles','credit_transactions','reports','anon_report_log',
    'stripe_events','support_messages','upgrade_intent'
  ] loop
    if exists (select from information_schema.tables
               where table_schema = 'public' and table_name = t) then
      execute format('drop policy if exists bi_readonly_select on public.%I', t);
      execute format(
        'create policy bi_readonly_select on public.%I for select to chartsage_readonly using (true)', t);
    end if;
  end loop;
end $$;

-- Done. Connect Evidence with this role (see bi/README.md):
--   Supabase → Project Settings → Database → Connection string → "Session pooler"
--   • Port 5432  (NOT the 6543 transaction pooler — Evidence uses server-side cursors)
--   • Session pooler user:  chartsage_readonly.<project-ref>
--     Direct connection user: chartsage_readonly
