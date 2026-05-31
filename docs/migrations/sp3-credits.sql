-- SP3 credits: tables, atomic functions, RLS.
-- Run once in Supabase → SQL editor.

create table if not exists profiles (
  user_id         uuid primary key references auth.users(id),
  credits_balance int  not null default 0,
  created_at      timestamptz not null default now()
);

create table if not exists credit_transactions (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id),
  delta       int  not null,
  reason      text not null,
  ref         text,
  created_at  timestamptz not null default now()
);
create index if not exists credit_transactions_user_idx
  on credit_transactions (user_id, created_at desc);

create table if not exists upgrade_intent (
  user_id    uuid primary key references auth.users(id),
  email      text,
  created_at timestamptz not null default now()
);

create or replace function spend_credits(p_user uuid, p_amount int, p_reason text, p_ref text default null)
returns int language plpgsql as $$
declare new_balance int;
begin
  update profiles set credits_balance = credits_balance - p_amount
    where user_id = p_user and credits_balance >= p_amount
    returning credits_balance into new_balance;
  if new_balance is null then
    raise exception 'INSUFFICIENT_CREDITS';
  end if;
  insert into credit_transactions(user_id, delta, reason, ref)
    values (p_user, -p_amount, p_reason, p_ref);
  return new_balance;
end; $$;

create or replace function grant_credits(p_user uuid, p_amount int, p_reason text, p_ref text default null)
returns int language plpgsql as $$
declare new_balance int;
begin
  update profiles set credits_balance = credits_balance + p_amount
    where user_id = p_user
    returning credits_balance into new_balance;
  insert into credit_transactions(user_id, delta, reason, ref)
    values (p_user, p_amount, p_reason, p_ref);
  return new_balance;
end; $$;

create or replace function ensure_profile(p_user uuid, p_grant int)
returns int language plpgsql as $$
declare new_balance int;
begin
  insert into profiles(user_id, credits_balance) values (p_user, 0)
    on conflict (user_id) do nothing;
  if not exists (select 1 from credit_transactions
                 where user_id = p_user and reason = 'signup_grant') then
    perform grant_credits(p_user, p_grant, 'signup_grant', null);
  end if;
  select credits_balance into new_balance from profiles where user_id = p_user;
  return new_balance;
end; $$;

alter table profiles enable row level security;
alter table credit_transactions enable row level security;
alter table upgrade_intent enable row level security;
create policy profiles_owner_select on profiles for select using (auth.uid() = user_id);
create policy txns_owner_select on credit_transactions for select using (auth.uid() = user_id);
