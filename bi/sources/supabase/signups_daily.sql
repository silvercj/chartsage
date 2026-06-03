-- Registered users per day (one profiles row is created per signup).
select
  date_trunc('day', created_at)::date as day,
  count(*) as signups
from profiles
group by 1
order by 1
