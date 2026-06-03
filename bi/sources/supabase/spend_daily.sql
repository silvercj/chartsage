-- Credits spent per day (report generation etc.). Spends are negative deltas,
-- so negate the sum to report a positive "credits spent" figure.
select
  date_trunc('day', created_at)::date as day,
  count(*)                            as spend_events,
  -sum(delta)                         as credits_spent
from credit_transactions
where delta < 0
group by 1
order by 1
