# ChartSage — Product & Model Health dashboard (PostHog, Terraform)
# ------------------------------------------------------------------
# Provider: posthog v1.0.6 (assumes you already have the `posthog` provider
#   block configured — this file only declares the dashboard + insights).
# Apply: `terraform apply` creates a fresh dashboard with all tiles below.
#
# Notes
# - Insights are HogQL (DataVisualizationNode + HogQLQuery) so they're robust to
#   schema drift and easy to tweak. Change a `display` to ActionsBar / ActionsLineGraph
#   / ActionsTable / ActionsPie in the UI any time.
# - The "model quality" tiles read `report_charts_composed` (added 2026-06-06) — they
#   stay empty until new reports are generated, then fill in. See docs/analytics-events.md.
# - All resources are tagged managed-by:terraform.

resource "posthog_dashboard" "chartsage" {
  name   = "ChartSage — Product & Model Health"
  pinned = true
  tags   = ["managed-by:terraform"]
}

# ── At a glance ───────────────────────────────────────────────────────────────

resource "posthog_insight" "summary_30d" {
  name = "Last 30 days — at a glance"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT countIf(event = 'report_generation_succeeded') AS reports, countIf(event = 'report_published') AS published, countIf(event = 'report_generation_failed') AS failed, round(100 * countIf(event = 'report_charts_composed' AND toFloat(properties.fallbackChartCount) > 0) / nullIf(countIf(event = 'report_charts_composed'), 0), 1) AS fallback_rate_pct, round(sumIf(toFloat(properties.estCostUsd), event = 'report_generation_succeeded'), 2) AS model_spend_usd FROM events WHERE timestamp > now() - INTERVAL 30 DAY"
    },
    "display" : "ActionsTable"
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Generation volume & health ────────────────────────────────────────────────

resource "posthog_insight" "reports_per_day" {
  name = "Reports generated per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, count() AS reports, count(DISTINCT person_id) AS users FROM events WHERE event = 'report_generation_succeeded' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "reports" }, { "column" : "users" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "generation_outcomes" {
  name = "Generation outcomes per day (started / succeeded / failed)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'report_generation_started') AS started, countIf(event = 'report_generation_succeeded') AS succeeded, countIf(event = 'report_generation_failed') AS failed FROM events WHERE event IN ('report_generation_started', 'report_generation_succeeded', 'report_generation_failed') GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "started" }, { "column" : "succeeded" }, { "column" : "failed" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Model output quality (the fallback story) ─────────────────────────────────

resource "posthog_insight" "fallback_rate_daily" {
  name = "Chart fallback rate per day (%)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, round(100 * countIf(toFloat(properties.fallbackChartCount) > 0) / count(), 1) AS any_fallback_pct, round(100 * countIf(toFloat(properties.fallbackRatio) >= 1) / count(), 1) AS all_fallback_pct FROM events WHERE event = 'report_charts_composed' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsLineGraph",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "any_fallback_pct" }, { "column" : "all_fallback_pct" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "fallback_outcomes_daily" {
  name = "Reports by chart-selection outcome per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, countIf(toFloat(properties.fallbackChartCount) = 0) AS clean, countIf(toFloat(properties.fallbackChartCount) > 0 AND toFloat(properties.fallbackRatio) < 1) AS partial_fallback, countIf(toFloat(properties.fallbackRatio) >= 1) AS all_fallback FROM events WHERE event = 'report_charts_composed' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "clean" }, { "column" : "partial_fallback" }, { "column" : "all_fallback" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "charts_model_vs_fallback" {
  name = "Avg charts per report — model vs fallback"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, round(avg(toFloat(properties.modelChartCount)), 2) AS avg_model_charts, round(avg(toFloat(properties.fallbackChartCount)), 2) AS avg_fallback_charts FROM events WHERE event = 'report_charts_composed' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "avg_model_charts" }, { "column" : "avg_fallback_charts" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "fallback_by_dataset_size" {
  name = "Fallback rate by dataset size (where the model under-selects)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT multiIf(toFloat(properties.rowCount) < 20, 'a. <20 rows', toFloat(properties.rowCount) < 100, 'b. 20-99 rows', toFloat(properties.rowCount) < 1000, 'c. 100-999 rows', 'd. 1000+ rows') AS dataset_size, count() AS reports, round(100 * countIf(toFloat(properties.fallbackChartCount) > 0) / count(), 1) AS fallback_rate_pct FROM events WHERE event = 'report_charts_composed' GROUP BY dataset_size ORDER BY dataset_size"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "dataset_size" }, "yAxis" : [{ "column" : "fallback_rate_pct" }, { "column" : "reports" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Performance & cost ────────────────────────────────────────────────────────

resource "posthog_insight" "generation_latency" {
  name = "Generation latency p50 / p95 (ms)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, round(quantile(0.5)(toFloat(properties.elapsedMs))) AS p50_ms, round(quantile(0.95)(toFloat(properties.elapsedMs))) AS p95_ms FROM events WHERE event = 'report_generation_succeeded' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsLineGraph",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "p50_ms" }, { "column" : "p95_ms" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "model_spend_daily" {
  name = "Model spend per day (USD)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, round(sum(toFloat(properties.estCostUsd)), 2) AS spend_usd, round(avg(toFloat(properties.estCostUsd)), 4) AS avg_per_report FROM events WHERE event = 'report_generation_succeeded' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "spend_usd" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Engagement ────────────────────────────────────────────────────────────────

resource "posthog_insight" "reports_published_daily" {
  name = "Reports published per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, count() AS published, count(DISTINCT person_id) AS users FROM events WHERE event = 'report_published' GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "published" }, { "column" : "users" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "exports_by_format" {
  name = "Exports by format"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT properties.format AS format, count() AS exports FROM events WHERE event = 'report_exported' GROUP BY format ORDER BY exports DESC"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "format" }, "yAxis" : [{ "column" : "exports" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "report_feedback" {
  name = "Report feedback (thumbs)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toString(properties.rating) AS rating, count() AS votes FROM events WHERE event = 'report_feedback' GROUP BY rating ORDER BY votes DESC"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "rating" }, "yAxis" : [{ "column" : "votes" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "post_gen_actions" {
  name = "Post-generation actions per day (generate-more / deepen / add-chart)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'generate_more_succeeded') AS generate_more, countIf(event = 'deepen_succeeded') AS deepen, countIf(event = 'add_chart_succeeded') AS add_chart FROM events WHERE event IN ('generate_more_succeeded', 'deepen_succeeded', 'add_chart_succeeded') GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "generate_more" }, { "column" : "deepen" }, { "column" : "add_chart" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Growth & monetisation ─────────────────────────────────────────────────────

resource "posthog_insight" "signups_logins" {
  name = "New signups & logins per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'credits_granted') AS new_signups, countIf(event = 'logged_in') AS logins FROM events WHERE event IN ('credits_granted', 'logged_in') GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "new_signups" }, { "column" : "logins" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "credits_funnel" {
  name = "Credits friction & purchase funnel per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'out_of_credits') AS out_of_credits, countIf(event = 'buy_pack_clicked') AS buy_clicks, countIf(event = 'checkout_started') AS checkouts_started FROM events WHERE event IN ('out_of_credits', 'buy_pack_clicked', 'checkout_started') GROUP BY day ORDER BY day"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "day" }, "yAxis" : [{ "column" : "out_of_credits" }, { "column" : "buy_clicks" }, { "column" : "checkouts_started" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "marketing_cta" {
  name = "Marketing CTA clicks by location"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : {
      "kind" : "HogQLQuery",
      "query" : "SELECT properties.location AS location, count() AS clicks FROM events WHERE event = 'marketing_cta_clicked' GROUP BY location ORDER BY clicks DESC"
    },
    "display" : "ActionsBar",
    "chartSettings" : { "xAxis" : { "column" : "location" }, "yAxis" : [{ "column" : "clicks" }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Layout (tile order top-to-bottom) ─────────────────────────────────────────

resource "posthog_dashboard_layout" "chartsage" {
  dashboard_id = posthog_dashboard.chartsage.id
  tiles = [
    { insight_id = posthog_insight.summary_30d.id },
    { insight_id = posthog_insight.reports_per_day.id },
    { insight_id = posthog_insight.generation_outcomes.id },
    { insight_id = posthog_insight.fallback_rate_daily.id },
    { insight_id = posthog_insight.fallback_outcomes_daily.id },
    { insight_id = posthog_insight.charts_model_vs_fallback.id },
    { insight_id = posthog_insight.fallback_by_dataset_size.id },
    { insight_id = posthog_insight.generation_latency.id },
    { insight_id = posthog_insight.model_spend_daily.id },
    { insight_id = posthog_insight.reports_published_daily.id },
    { insight_id = posthog_insight.exports_by_format.id },
    { insight_id = posthog_insight.report_feedback.id },
    { insight_id = posthog_insight.post_gen_actions.id },
    { insight_id = posthog_insight.signups_logins.id },
    { insight_id = posthog_insight.credits_funnel.id },
    { insight_id = posthog_insight.marketing_cta.id },
  ]
}
