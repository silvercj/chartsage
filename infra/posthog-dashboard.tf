# ChartSage — Product & Model Health dashboard (PostHog, Terraform)
# ------------------------------------------------------------------
# Provider: posthog v1.0.6 (declare the `posthog` provider block separately).
# Apply: `terraform apply` creates the dashboard + all tiles.
#
# Design system (semantic colour — readable at a glance):
#   • red/amber  = problems (failures, fallback, friction)
#   • green      = healthy (success, published, clean)
#   • indigo     = volume / neutral
# Every multi-series tile has a legend (showLegend) + human labels; rates show %,
# cost shows $, latency shows ms; the fallback-rate tile carries a target goal line.
# Model-quality tiles read `report_charts_composed` (live 2026-06-06) — they fill
# as new reports run. See docs/analytics-events.md.

locals {
  c_good    = "#10b981" # emerald  — success / healthy
  c_bad     = "#ef4444" # red      — failures / total fallback
  c_warn    = "#f59e0b" # amber    — partial fallback / latency tail / friction
  c_primary = "#6366f1" # indigo   — volume / neutral
  c_accent  = "#14b8a6" # teal     — secondary series
  c_violet  = "#8b5cf6" # violet   — tertiary series
}

resource "posthog_dashboard" "chartsage" {
  name   = "ChartSage — Product & Model Health"
  pinned = true
  tags   = ["managed-by:terraform"]
}

# ── Headline numbers (last 30 days) ───────────────────────────────────────────

resource "posthog_insight" "kpi_reports_30d" {
  name = "Reports generated · 30d"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT count() AS reports FROM events WHERE event = 'report_generation_succeeded' AND timestamp > now() - INTERVAL 30 DAY" },
    "display" : "BoldNumber",
    "chartSettings" : { "yAxis" : [{ "column" : "reports", "settings" : { "formatting" : { "style" : "short", "decimalPlaces" : 0 }, "display" : { "color" : local.c_primary } } }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "kpi_fallback_rate_30d" {
  name = "Chart fallback rate · 30d"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT round(100 * countIf(toFloat(properties.fallbackChartCount) > 0) / count(), 1) AS fallback_rate FROM events WHERE event = 'report_charts_composed' AND timestamp > now() - INTERVAL 30 DAY" },
    "display" : "BoldNumber",
    "chartSettings" : { "yAxis" : [{ "column" : "fallback_rate", "settings" : { "formatting" : { "suffix" : "%", "decimalPlaces" : 1 }, "display" : { "color" : local.c_warn } } }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "kpi_spend_30d" {
  name = "Model spend · 30d"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT round(sum(toFloat(properties.estCostUsd)), 2) AS spend FROM events WHERE event = 'report_generation_succeeded' AND timestamp > now() - INTERVAL 30 DAY" },
    "display" : "BoldNumber",
    "chartSettings" : { "yAxis" : [{ "column" : "spend", "settings" : { "formatting" : { "prefix" : "$", "decimalPlaces" : 2 }, "display" : { "color" : local.c_good } } }] }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Model output quality (the fallback story) ─────────────────────────────────

resource "posthog_insight" "fallback_rate_daily" {
  name = "Chart fallback rate per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, round(100 * countIf(toFloat(properties.fallbackChartCount) > 0) / count(), 1) AS any_fallback, round(100 * countIf(toFloat(properties.fallbackRatio) >= 1) / count(), 1) AS all_fallback FROM events WHERE event = 'report_charts_composed' GROUP BY day ORDER BY day" },
    "display" : "ActionsLineGraph",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "any_fallback", "settings" : { "formatting" : { "suffix" : "%", "decimalPlaces" : 1 }, "display" : { "label" : "Any fallback", "color" : local.c_warn } } },
        { "column" : "all_fallback", "settings" : { "formatting" : { "suffix" : "%", "decimalPlaces" : 1 }, "display" : { "label" : "All charts fallback", "color" : local.c_bad } } }
      ],
      "showLegend" : true,
      "goalLines" : [{ "label" : "Target ≤ 10%", "value" : 10, "displayLabel" : true }]
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "fallback_outcomes_daily" {
  name = "Reports by chart-selection outcome per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, countIf(toFloat(properties.fallbackChartCount) = 0) AS clean, countIf(toFloat(properties.fallbackChartCount) > 0 AND toFloat(properties.fallbackRatio) < 1) AS partial, countIf(toFloat(properties.fallbackRatio) >= 1) AS total_miss FROM events WHERE event = 'report_charts_composed' GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "clean", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Model-selected (clean)", "color" : local.c_good } } },
        { "column" : "partial", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Partial fallback", "color" : local.c_warn } } },
        { "column" : "total_miss", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "All fallback", "color" : local.c_bad } } }
      ],
      "showLegend" : true, "stackBars100" : false
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "charts_model_vs_fallback" {
  name = "Avg charts per report — model vs fallback"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, round(avg(toFloat(properties.modelChartCount)), 2) AS model_charts, round(avg(toFloat(properties.fallbackChartCount)), 2) AS fallback_charts FROM events WHERE event = 'report_charts_composed' GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "model_charts", "settings" : { "formatting" : { "decimalPlaces" : 2 }, "display" : { "label" : "Model-selected", "color" : local.c_good } } },
        { "column" : "fallback_charts", "settings" : { "formatting" : { "decimalPlaces" : 2 }, "display" : { "label" : "Fallback", "color" : local.c_warn } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "fallback_by_dataset_size" {
  name = "Fallback rate by dataset size"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT multiIf(toFloat(properties.rowCount) < 20, 'a. <20 rows', toFloat(properties.rowCount) < 100, 'b. 20-99 rows', toFloat(properties.rowCount) < 1000, 'c. 100-999 rows', 'd. 1000+ rows') AS dataset_size, round(100 * countIf(toFloat(properties.fallbackChartCount) > 0) / count(), 1) AS fallback_rate, count() AS reports FROM events WHERE event = 'report_charts_composed' GROUP BY dataset_size ORDER BY dataset_size" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "dataset_size" },
      "yAxis" : [
        { "column" : "fallback_rate", "settings" : { "formatting" : { "suffix" : "%", "decimalPlaces" : 1 }, "display" : { "label" : "Fallback rate", "color" : local.c_bad, "yAxisPosition" : "left" } } },
        { "column" : "reports", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Reports", "color" : local.c_primary, "yAxisPosition" : "right" } } }
      ],
      "showLegend" : true, "showValuesOnSeries" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Generation volume & health ────────────────────────────────────────────────

resource "posthog_insight" "reports_per_day" {
  name = "Reports generated per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, count() AS reports, count(DISTINCT person_id) AS users FROM events WHERE event = 'report_generation_succeeded' GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "reports", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Reports", "color" : local.c_primary } } },
        { "column" : "users", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Distinct users", "color" : local.c_accent } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "generation_outcomes" {
  name = "Generation outcomes per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'report_generation_started') AS started, countIf(event = 'report_generation_succeeded') AS succeeded, countIf(event = 'report_generation_failed') AS failed FROM events WHERE event IN ('report_generation_started', 'report_generation_succeeded', 'report_generation_failed') GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "started", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Started", "color" : local.c_primary } } },
        { "column" : "succeeded", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Succeeded", "color" : local.c_good } } },
        { "column" : "failed", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Failed", "color" : local.c_bad } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "generation_latency" {
  name = "Generation latency p50 / p95"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, round(quantile(0.5)(toFloat(properties.elapsedMs))) AS p50, round(quantile(0.95)(toFloat(properties.elapsedMs))) AS p95 FROM events WHERE event = 'report_generation_succeeded' GROUP BY day ORDER BY day" },
    "display" : "ActionsLineGraph",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "p50", "settings" : { "formatting" : { "suffix" : " ms", "decimalPlaces" : 0 }, "display" : { "label" : "p50", "color" : local.c_primary } } },
        { "column" : "p95", "settings" : { "formatting" : { "suffix" : " ms", "decimalPlaces" : 0 }, "display" : { "label" : "p95", "color" : local.c_warn } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "model_spend_daily" {
  name = "Model spend per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, round(sum(toFloat(properties.estCostUsd)), 2) AS spend FROM events WHERE event = 'report_generation_succeeded' GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [{ "column" : "spend", "settings" : { "formatting" : { "prefix" : "$", "decimalPlaces" : 2 }, "display" : { "label" : "Spend", "color" : local.c_good } } }]
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Acquisition & conversion ──────────────────────────────────────────────────

resource "posthog_insight" "unique_visitors_daily" {
  name = "Unique visitors & pageviews per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, count(DISTINCT person_id) AS unique_visitors, count() AS pageviews FROM events WHERE event = '$pageview' GROUP BY day ORDER BY day" },
    "display" : "ActionsLineGraph",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "unique_visitors", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Unique visitors", "color" : local.c_primary } } },
        { "column" : "pageviews", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Pageviews", "color" : local.c_accent } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "activation_funnel" {
  name = "Activation funnel — visit -> generate -> publish (90d)"
  query_json = jsonencode({
    "kind" : "InsightVizNode",
    "source" : {
      "kind" : "FunnelsQuery",
      "series" : [
        { "kind" : "EventsNode", "event" : "$pageview", "name" : "Visited" },
        { "kind" : "EventsNode", "event" : "report_generation_succeeded", "name" : "Generated a report" },
        { "kind" : "EventsNode", "event" : "report_published", "name" : "Published" }
      ],
      "funnelsFilter" : { "funnelVizType" : "steps", "funnelOrderType" : "ordered" },
      "dateRange" : { "date_from" : "-90d" }
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "traffic_by_utm_source" {
  name = "Traffic by UTM source"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT coalesce(properties.utm_source, '(direct / none)') AS source, count(DISTINCT person_id) AS visitors, count() AS pageviews FROM events WHERE event = '$pageview' GROUP BY source ORDER BY visitors DESC LIMIT 15" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "source" },
      "yAxis" : [
        { "column" : "visitors", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Visitors", "color" : local.c_primary } } },
        { "column" : "pageviews", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Pageviews", "color" : local.c_accent } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "x_posts_by_campaign" {
  name = "X posts - traffic by campaign (utm_source = x)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT coalesce(properties.utm_campaign, '(none)') AS campaign, count(DISTINCT person_id) AS visitors, count() AS pageviews FROM events WHERE event = '$pageview' AND properties.utm_source = 'x' GROUP BY campaign ORDER BY visitors DESC" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "campaign" },
      "yAxis" : [{ "column" : "visitors", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Visitors", "color" : local.c_violet } } }],
      "showValuesOnSeries" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Engagement ────────────────────────────────────────────────────────────────

resource "posthog_insight" "reports_published_daily" {
  name = "Reports published per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, count() AS published, count(DISTINCT person_id) AS users FROM events WHERE event = 'report_published' GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "published", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Published", "color" : local.c_good } } },
        { "column" : "users", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Distinct users", "color" : local.c_accent } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "post_gen_actions" {
  name = "Post-generation actions per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'generate_more_succeeded') AS generate_more, countIf(event = 'deepen_succeeded') AS deepen, countIf(event = 'add_chart_succeeded') AS add_chart FROM events WHERE event IN ('generate_more_succeeded', 'deepen_succeeded', 'add_chart_succeeded') GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "generate_more", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Generate more", "color" : local.c_primary } } },
        { "column" : "deepen", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Deepen", "color" : local.c_violet } } },
        { "column" : "add_chart", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Add chart", "color" : local.c_accent } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "exports_by_format" {
  name = "Exports by format"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT properties.format AS format, count() AS exports FROM events WHERE event = 'report_exported' GROUP BY format ORDER BY exports DESC" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "format" },
      "yAxis" : [{ "column" : "exports", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Exports", "color" : local.c_primary } } }],
      "showValuesOnSeries" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "report_feedback" {
  name = "Report feedback (thumbs)"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toString(properties.rating) AS rating, count() AS votes FROM events WHERE event = 'report_feedback' GROUP BY rating ORDER BY votes DESC" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "rating" },
      "yAxis" : [{ "column" : "votes", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Votes", "color" : local.c_good } } }],
      "showValuesOnSeries" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Growth & monetisation ─────────────────────────────────────────────────────

resource "posthog_insight" "signups_logins" {
  name = "New signups & logins per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'credits_granted') AS new_signups, countIf(event = 'logged_in') AS logins FROM events WHERE event IN ('credits_granted', 'logged_in') GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "new_signups", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "New signups", "color" : local.c_good } } },
        { "column" : "logins", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Logins", "color" : local.c_primary } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "credits_funnel" {
  name = "Credits friction & purchase funnel per day"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT toDate(timestamp) AS day, countIf(event = 'out_of_credits') AS out_of_credits, countIf(event = 'buy_pack_clicked') AS buy_clicks, countIf(event = 'checkout_started') AS checkouts FROM events WHERE event IN ('out_of_credits', 'buy_pack_clicked', 'checkout_started') GROUP BY day ORDER BY day" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "day" },
      "yAxis" : [
        { "column" : "out_of_credits", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Hit out-of-credits", "color" : local.c_warn } } },
        { "column" : "buy_clicks", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Clicked buy", "color" : local.c_primary } } },
        { "column" : "checkouts", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Started checkout", "color" : local.c_good } } }
      ],
      "showLegend" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

resource "posthog_insight" "marketing_cta" {
  name = "Marketing CTA clicks by location"
  query_json = jsonencode({
    "kind" : "DataVisualizationNode",
    "source" : { "kind" : "HogQLQuery", "query" : "SELECT properties.location AS location, count() AS clicks FROM events WHERE event = 'marketing_cta_clicked' GROUP BY location ORDER BY clicks DESC" },
    "display" : "ActionsBar",
    "chartSettings" : {
      "xAxis" : { "column" : "location" },
      "yAxis" : [{ "column" : "clicks", "settings" : { "formatting" : { "decimalPlaces" : 0 }, "display" : { "label" : "Clicks", "color" : local.c_violet } } }],
      "showValuesOnSeries" : true
    }
  })
  tags          = ["managed-by:terraform"]
  dashboard_ids = [posthog_dashboard.chartsage.id]
}

# ── Layout (top-to-bottom: headlines → model quality → volume → engagement → growth)

resource "posthog_dashboard_layout" "chartsage" {
  dashboard_id = posthog_dashboard.chartsage.id
  tiles = [
    { insight_id = posthog_insight.kpi_reports_30d.id },
    { insight_id = posthog_insight.kpi_fallback_rate_30d.id },
    { insight_id = posthog_insight.kpi_spend_30d.id },
    { insight_id = posthog_insight.fallback_rate_daily.id },
    { insight_id = posthog_insight.fallback_outcomes_daily.id },
    { insight_id = posthog_insight.charts_model_vs_fallback.id },
    { insight_id = posthog_insight.fallback_by_dataset_size.id },
    { insight_id = posthog_insight.reports_per_day.id },
    { insight_id = posthog_insight.generation_outcomes.id },
    { insight_id = posthog_insight.generation_latency.id },
    { insight_id = posthog_insight.model_spend_daily.id },
    { insight_id = posthog_insight.unique_visitors_daily.id },
    { insight_id = posthog_insight.activation_funnel.id },
    { insight_id = posthog_insight.traffic_by_utm_source.id },
    { insight_id = posthog_insight.x_posts_by_campaign.id },
    { insight_id = posthog_insight.reports_published_daily.id },
    { insight_id = posthog_insight.post_gen_actions.id },
    { insight_id = posthog_insight.exports_by_format.id },
    { insight_id = posthog_insight.report_feedback.id },
    { insight_id = posthog_insight.signups_logins.id },
    { insight_id = posthog_insight.credits_funnel.id },
    { insight_id = posthog_insight.marketing_cta.id },
  ]
}
