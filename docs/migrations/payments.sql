-- docs/migrations/payments.sql
-- SP4 payments: Stripe event ledger + atomic, idempotent purchase processing.
-- Run once in Supabase → SQL editor (after sp3-credits.sql).

create table if not exists stripe_events (
  event_id   text primary key,            -- Stripe event id (evt_…); the idempotency key
  created_at timestamptz not null default now()
);
alter table stripe_events enable row level security;  -- service-role only; no client policies

-- Atomic: record the event and grant in a single transaction. Returns
-- {granted, balance}. On a duplicate event_id it grants nothing
-- (granted=false) and returns the current balance, so the webhook can fire
-- analytics exactly once. Reuses grant_credits() for the ledger entry
-- (reason 'stripe_purchase', ref = checkout session id).
create or replace function process_stripe_purchase(
  p_event text, p_user uuid, p_credits int, p_ref text
) returns jsonb language plpgsql as $$
declare new_balance int; n_inserted int;
begin
  insert into stripe_events(event_id) values (p_event)
    on conflict (event_id) do nothing;
  get diagnostics n_inserted = row_count;       -- 1 = new, 0 = duplicate
  if n_inserted = 0 then
    select credits_balance into new_balance from profiles where user_id = p_user;
    return jsonb_build_object('granted', false, 'balance', coalesce(new_balance, 0));
  end if;
  insert into profiles(user_id, credits_balance) values (p_user, 0)
    on conflict (user_id) do nothing;           -- safety: profile must exist to credit
  perform grant_credits(p_user, p_credits, 'stripe_purchase', p_ref);
  select credits_balance into new_balance from profiles where user_id = p_user;
  return jsonb_build_object('granted', true, 'balance', new_balance);
end; $$;
