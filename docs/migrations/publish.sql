-- docs/migrations/publish.sql
-- SP5-A publish/shareable reports. Run once in Supabase (the `reports` table
-- already exists from SP1). Opt-in public visibility + OG image key.
alter table reports add column if not exists is_public    boolean not null default false;
alter table reports add column if not exists og_image_key text;
alter table reports add column if not exists published_at  timestamptz;
create index if not exists reports_public_idx
  on reports (is_public, updated_at desc) where is_public;
