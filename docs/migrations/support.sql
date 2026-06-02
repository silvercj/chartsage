-- Beta: support contact messages. Run once in the Supabase SQL editor.
create table if not exists support_messages (
  id         uuid primary key default gen_random_uuid(),
  email      text,
  message    text not null,
  user_id    uuid,
  anon_id    uuid,
  created_at timestamptz not null default now()
);
create index if not exists support_messages_created_idx on support_messages (created_at desc);
alter table support_messages enable row level security;  -- service-role only; no client policies
