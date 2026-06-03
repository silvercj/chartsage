# ChartSage Social Playbook — X + Reddit (channel 2 execution)

*The operational layer under [marketing-strategy.md](marketing-strategy.md). The strategy concluded growth is organic/product-led; this is how we actually run the social-distribution channel, week to week. Created 2026-06-03.*

---

## The two decisions that frame everything

1. **Tooling is free.** Buffer's free tier schedules X at $0 (it absorbs X's new per-connection API fee). We are **not** paying $22/mo for Metricool — it costs ~$25–27/mo once the forced +$5 X add-on is included, and it can't post to Reddit anyway. Time is the input, not money.
2. **Paid ads are a *validation experiment*, not an acquisition channel.** At ~£25/week we cannot read a clean conversion signal or let any ad-platform optimiser work (that needs ~$1,500–2,000/mo). And a $5–40 *one-time* product with ~$4–12 margin and no recurring revenue **cannot make cold paid pay back at any budget** — so we don't pretend it's a growth engine. What ~£15–25/week *can* buy: a **directional read on whether strangers find this useful and whether anyone pulls toward paying**, plus a seeded **retargeting audience**. That's the entire purpose of the ad spend: learning, not scaling.

> The growth engine remains: compounding SEO/shareable reports (shipped) + this social channel + community. Paid is a measurement instrument bolted on the side.

---

## The loop (two tracks, one measurement layer)

```
                 you hand Claude datasets
                          │
                 Claude runs them through ChartSage,
                 ranks by "surprising / shareable",
                 publishes the 2–3 winners
                          │
        ┌─────────────────┴───────────────────┐
   TRACK A: X (broadcast)              TRACK B: Reddit (participate)
   showcase posts, scheduled          value-first; Claude hands you
   via Buffer (hero image + link)     full copy-paste text + image
        │                                      │
        └──────────────┬───────────────────────┘
                       │
            TRACK C: X ads (validation) → seeds retargeting pool
                       │
              PostHog funnel + session recordings
              (click → report → signup → repeat → pay)
                       │
            weekly 10-min review → double down / kill
```

---

## Setup — one-time (you)

- [ ] **Buffer Free** — sign up, connect the ChartSage **X** account. (Reddit is *not* scheduled — it's manual, see Track B.)
- [ ] **X Ads account** — create it; create a **conversion pixel**; send me the **pixel ID** (I install it — see the Build section). The pixel is what makes retargeting + conversion tracking possible.
- [ ] **UTM convention** — every social link uses:
  `?utm_source=x|reddit|x_ads&utm_medium=social&utm_campaign=<theme>&utm_content=<short-id>`
  PostHog auto-captures these, so the funnel segments by source with zero extra work.
- [ ] **PostHog** — already live; confirm session recordings are on (they're our best qualitative tool for "is it useful").

---

## Track A — X showcase engine (agent-run)

**Division of labour:** *you* supply datasets (CSV files or links — Kaggle, data.gov, Our World in Data, FiveThirtyEight, a topical news CSV). *I* run them through ChartSage, read every result, and pick the 2–3 with a genuinely **surprising, screenshot-worthy** finding. For each winner, the report gets generated + published in the app (I drive it via the API, or you click-generate just the handful I pick), and **I draft the post**: caption + the report's hero image + the UTM link. You approve; Buffer schedules it.

**The repeatable post (template bank — I'll tailor each one):**
> *"Ran [the 16k video-game-sales dataset] through ChartSage. Didn't expect this: [one surprising finding in a sentence]. Full interactive report 👇"* + hero image + link

Variants I'll rotate: the surprising-stat hook, the "I asked a question of the data" hook, the "here's a chart most people get wrong" hook, the before/after ("a CSV nobody could read → this").

**Cadence:** 3–5×/week, best-times via Buffer. Consistency > volume.

**Dataset wells to keep the showcases flowing:** Kaggle "trending/hot" datasets, Our World in Data, data.gov, FiveThirtyEight's data repo, Google Dataset Search, topical/seasonal news data. Hand me a batch and I'll mine them.

---

## Track B — Reddit (value-first, *carefully*)

**Reddit is savage about anything that looks like advertising — and it's right to be.** Get this wrong and the account is shadowbanned, the post removed, and the brand burned. So the entire posture here is: **you are a person who is genuinely helpful and happens to have built a tool — not a brand broadcasting.**

**What Claude delivers for Reddit:** the **full, copy-paste-ready text** of every comment/post, **plus the actual image** (e.g. a genuine OC chart for r/dataisbeautiful). Never a vague "go comment on this" — always the literal words + the asset, matched to the specific thread.

**The 90/10 rule (hard):** ≥90% of your Reddit activity is pure help with **no link and no mention**. ≤10% may mention ChartSage — and *only* where (a) the subreddit's rules permit it, (b) it genuinely answers the question, and (c) you've already given a complete answer without it.

**The anti-advertising guardrails (the part you flagged):**
- **Never repeat-post the same showcase across subreddits.** Reddit's spam filters and mods catch identical/near-identical link posts instantly. Each contribution is bespoke to its thread.
- **Never lead with the link.** Lead with a real answer. The tool, if mentioned at all, is a footnote ("fwiw I built a thing that does this, but here's how to do it manually too").
- **Use your personal Reddit account, not a "ChartSage" brand account** (confirmed). Reddit distrusts brand accounts doing promo; participating as a real person is far safer. X stays the brand account; Reddit is you.
- **Account hygiene:** aged account + real karma from genuine participation before *any* mention. New accounts dropping links get auto-removed.
- **Read each subreddit's rules every time.** Many ban self-promo outright; some have a weekly self-promo thread; r/dataisbeautiful requires OC + a source-data comment and forbids tool-promotion in the post.
- **Space mentions out** (weeks apart, different subs) so there's no detectable "advertising" pattern.
- **When in doubt, don't link.** A helpful comment with zero promotion still builds presence and is never punished.

**Subreddit map:**
| Subreddit | Play | Link allowed? |
|---|---|---|
| r/dataisbeautiful | Share a genuinely interesting **OC** chart as the post; tool only in a comment if asked + rules allow | Post: no promo. Comment: sparingly |
| r/excel, r/analytics, r/datascience | **Answer** "how do I chart/analyse X" questions; complete answer first, tool as one option | Rarely, only if earned |
| r/smallbusiness, r/entrepreneur | Value-first answers on "how do I understand my data/sales" | Rarely |
| r/SideProject, r/IMadeThis, r/InternetIsBeautiful | These **welcome** maker self-promo — the right place to actually share the product | Yes (per rules) |

**The honest near-term read:** Reddit is high-effort and slow, but it's where high-intent users actually are. Treat it as 2–3 *genuinely helpful* contributions/week, not a broadcast channel.

---

## Track C — X ads: the validation experiment

**The question:** *Do cold strangers find ChartSage useful enough to use it, and is there any pull toward paying?*

**Funnel nuance that shapes the read:** the first report is free with no signup, so a cold click can hit the full "wow" instantly. That means:
- **Activation ("did they generate a report?") is the fast, high-signal validation.** Target **>7–10%** of ad clickers.
- **Payment is a slow signal** — new users get 300 free credits (~3 reports), so willingness-to-pay reveals itself over *weeks*, not days. Don't expect purchases in week 1; expect to learn *usefulness* fast and *willingness-to-pay* slowly.

**Setup:** 2–3 creatives on the core value prop ("turn your spreadsheet into a report in seconds — no account"), interest targeting (Excel / spreadsheets / data analysis / "founder" interests), UTM `utm_source=x_ads`. Land on the homepage (or a tool landing page later).

**Budget:** **£25/week for ~4 weeks (~£100)** — the full weekly budget, freed up now the tool is £0. Still a learning budget: small, fixed, time-boxed.

**What we measure (PostHog, by `utm_source=x_ads`):**
- Activation rate (generated a report), signup rate, repeat-use, **any** purchases (even 1–2 = real demand), cost-per-activation.
- **Session recordings of ad clickers** — *where do they drop?* The single best "is it useful / where's the friction" tool we have.

**Kill / continue criteria (set NOW so we don't fool ourselves):**
- **Strong:** activation >10% **and** repeat-use / any purchases → demand + usefulness validated; invest in retargeting + a dedicated landing page.
- **Mixed:** activation 3–10%, little repeat use → useful-ish, but cold→value or cold→pay leap too big; fix activation/landing, keep organic primary.
- **Dead:** activation <3% → wrong audience or message (or product doesn't land cold); stop, rethink targeting/positioning before spending more.
- **Don't judge willingness-to-pay before ~60 days** (free credits delay it).

---

## Retargeting — the warm pool the ads build

Your insight is the smartest part of the paid plan: **the cold ad clickers who try it and bounce are the warmest audience you'll ever have** (retargeting converts 2–4× better and clicks 30–60% cheaper than cold). So the ad spend does double duty — validation signal **and** seeding a retargetable audience.

- The **X pixel** (installed via the Build below) auto-builds a "visited-but-didn't-convert" audience from day one.
- **Caveat:** X needs ~**100 people** in an audience before you can retarget it — so retargeting starts *after* the cold test has accumulated clickers. Install the pixel now precisely so that window isn't wasted.
- Retargeting creative: a gentle nudge — *"Still got that spreadsheet open? Your free report's waiting."*
- It's still margin-capped (one-time pricing), but this is validation-stage, and it's the one paid play with a plausible path to ROI.

---

## Measurement — the weekly ritual

PostHog already has the funnel. Once a week, ~10 minutes:
1. By `utm_source` (x, reddit, x_ads, organic): clicks → **report generated (activation)** → signup → repeat-use → purchase.
2. Benchmarks to beat: activation **>7–10%**, free→paid **5–15%** (industry), day-7 return (the survival metric for one-time pricing).
3. Which **post themes / datasets** drove activations? Make more of those; drop the duds.
4. Watch 2–3 session recordings of social/ad visitors — note friction.
5. **Don't over-react to small numbers.** Trends need ~60 days. Cut/scale a channel only on a real signal, and scale spend +20–30%/step from revenue — never double.

---

## 2-week starter sequence

**Week 1 — stand it up:**
- You: Buffer Free + connect X; create X Ads account + pixel, send me the pixel ID.
- Me: install the X pixel + conversion events (consent-gated); confirm UTM capture in PostHog.
- You: hand me the first **5 datasets**. Me: run them, pick 2–3 winners, publish, draft the first **6 X showcase posts** → you approve → Buffer schedules them across the week.
- Me: draft your first **2–3 Reddit contributions** (full text + image), matched to live threads I find; you post them as a human.

**Week 2 — first signal:**
- Launch the X ads validation test (£25/week, 2–3 creatives).
- Keep the showcase cadence (3–5 X posts); 2–3 more Reddit contributions.
- First weekly review: activation by source, session recordings, what's landing.

---

## Guardrails

- **Brand voice:** confident, plain-spoken, "show the deliverable, not the hype." Never overclaim.
- **Reddit:** the discipline above is non-negotiable — value-first, no repeat promo, human account, rules every time.
- **Disclosure:** where you mention the tool you built, say so plainly (it builds trust and follows platform/FTC norms).
- **Privacy:** the X pixel fires **only after cookie consent** (the existing banner). No tracking without consent.
- **No automation that posts without review** — everything is draft-and-approve at this stage.

---

## The one build (code) — X pixel + conversion tracking

*Small frontend change; gated on you providing the X pixel ID. Spec:*
- Add the **X (Twitter) pixel** base script to the app, **consent-gated** (only fires after the cookie banner's accept).
- Fire conversion events on the key funnel steps: `report_generated`, `signup`, `purchase` (so X can track conversions + build the retargeting audience). PostHog continues to own the primary funnel analytics; the pixel is purely for X's ad/retargeting layer.
- Verify in X Ads Manager that events register before spending.

---

## The "later" triggers (not now)

- **Scale retargeting** once the audience clears ~100 and the cold test shows any activation signal.
- **One small (~£100) fully-attributed paid test/month** only from **month 3+**, funded from revenue, once organic shows life — most likely warm retargeting, *not* cold ads.
- **Revisit a paid tool (Metricool/Buffer paid)** only if you outgrow free post caps or genuinely need deep competitor analytics.
- **Demo-clip / TikTok track** (the Julius playbook) if you later decide to produce short before/after video.
