---
title: ChartSage BI
---

Internal BI for ChartSage — built with [Evidence](https://evidence.dev), running **locally and free**. Numbers refresh whenever you run `npm run sources`.

## Product analytics — PostHog

_Unique users and event volume over the last 90 days, straight from the HogQL query API._

```sql ph_kpis
select
  max(case when step_order = 1 then users end) as visited,
  max(case when step_order = 3 then users end) as got_report,
  max(case when step_order = 4 then users end) as started_checkout
from posthog.funnel
```

<BigValue data={ph_kpis} value=visited title="Visited (90d)"/>
<BigValue data={ph_kpis} value=got_report title="Got a report"/>
<BigValue data={ph_kpis} value=started_checkout title="Started checkout"/>

### Activation funnel

```sql funnel
select step, users, step_order
from posthog.funnel
order by step_order
```

<BarChart
    data={funnel}
    x=step
    y=users
    swapXY=true
    sort=false
    title="Activation funnel — unique users (90d)"
    subtitle="Visited → started a report → got a report → started checkout"
/>

### Key events per day

```sql events_trend
select cast(day as date) as day, event, events
from posthog.events_daily
order by day
```

<LineChart
    data={events_trend}
    x=day
    y=events
    series=event
    title="Key product events per day (90d)"
/>

### All events (90d)

```sql top_events
select event, events, users
from posthog.event_totals
order by events desc
```

<DataTable data={top_events} rows=12 search=true>
    <Column id=event title="Event"/>
    <Column id=events title="Events" contentType=colorscale/>
    <Column id=users title="Unique users"/>
</DataTable>

## Business metrics — Supabase Postgres

_From the read-only Postgres role. **If these show errors,** do the one-time setup in `bi/README.md`:
run `docs/migrations/bi-readonly-role.sql` in Supabase, add the connection to `bi/.env`, then `npm run sources`._

```sql kpis
select
  (select coalesce(sum(signups), 0)           from supabase.signups_daily)  as signups,
  (select coalesce(sum(reports), 0)           from supabase.reports_daily)  as reports,
  (select coalesce(sum(purchases), 0)         from supabase.purchases_daily) as purchases,
  (select coalesce(sum(credits_purchased), 0) from supabase.purchases_daily) as credits_sold
```

<BigValue data={kpis} value=signups title="Signups"/>
<BigValue data={kpis} value=reports title="Reports"/>
<BigValue data={kpis} value=purchases title="Purchases"/>
<BigValue data={kpis} value=credits_sold title="Credits sold"/>

```sql conversion
select
  total_users,
  paying_users,
  case when total_users > 0 then paying_users::float / total_users else 0 end as conversion_rate
from supabase.conversion
```

<BigValue data={conversion} value=conversion_rate title="Free → Paid" fmt=pct1/>

### Signups & reports per day

```sql signups_daily
select day, signups from supabase.signups_daily order by day
```

<LineChart data={signups_daily} x=day y=signups title="Signups per day"/>

```sql reports_split
select day, 'Signed-in' as kind, signed_in_reports as n from supabase.reports_daily
union all
select day, 'Anonymous' as kind, anon_reports as n from supabase.reports_daily
order by day
```

<BarChart data={reports_split} x=day y=n series=kind type=stacked title="Reports per day — anonymous vs signed-in"/>

### Credits & conversion

```sql purchases_daily
select day, purchases, credits_purchased from supabase.purchases_daily order by day
```

<BarChart data={purchases_daily} x=day y=credits_purchased title="Credits purchased per day"/>

```sql spend_daily
select day, credits_spent from supabase.spend_daily order by day
```

<LineChart data={spend_daily} x=day y=credits_spent title="Credits spent per day"/>

### Anonymous activity

```sql anon_daily
select day, anon_attempts from supabase.anon_reports_daily order by day
```

<LineChart data={anon_daily} x=day y=anon_attempts title="Anonymous report attempts per day"/>

