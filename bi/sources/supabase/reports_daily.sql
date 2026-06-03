-- Reports generated per day, split by signed-in vs anonymous.
-- A signed-in report has user_id set; an anonymous one does not.
select
  date_trunc('day', created_at)::date as day,
  count(*)                              as reports,
  count(user_id)                        as signed_in_reports,
  count(*) - count(user_id)             as anon_reports
from reports
group by 1
order by 1
