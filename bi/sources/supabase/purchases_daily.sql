-- Stripe credit-pack purchases per day.
-- NB: only the credit `delta` is stored here, not the $ amount — so this tracks
-- purchase COUNT and CREDITS sold. True revenue lives in Stripe / PostHog.
select
  date_trunc('day', created_at)::date as day,
  count(*)                            as purchases,
  sum(delta)                          as credits_purchased
from credit_transactions
where reason = 'stripe_purchase'
group by 1
order by 1
