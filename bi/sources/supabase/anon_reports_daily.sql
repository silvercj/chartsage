-- Anonymous report attempts per day (pre-signup activity & abuse/cap view).
-- This is the raw anon attempt log, distinct from successful `reports`.
select
  date_trunc('day', created_at)::date as day,
  count(*) as anon_attempts
from anon_report_log
group by 1
order by 1
