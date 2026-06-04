# The X (Twitter) Algorithm — Reference & ChartSage Playbook

*Compiled 2026-06-04 from X/xAI's open-sourced code + community rundowns. Companion to [marketing-social-playbook.md](marketing-social-playbook.md). This is a living reference — re-verify before any big betting on a specific number; X changes this often and many "weights" floating around are from the 2023 legacy code, not the current model.*

---

## 0. TL;DR (if you read nothing else)

1. **Conversations beat broadcasts.** A reply that *you (the author) reply back to* is the single strongest signal — worth ~**150× a like**. Replies > reposts > likes, by a lot.
2. **External links get throttled** (~30–50%+ reach cut). **Put the link in the first reply, not the main post.**
3. **Text-first.** Native text/images outperform; don't default to video. A strong one-liner + one chart image beats a link-heavy post.
4. **Negative signals are nuclear.** One block/mute/report/"not interested" outweighs dozens of likes. Don't be annoying, clickbaity, or spammy.
5. **The first 30–60 min matter most.** Early engagement (esp. replies) decides whether it escalates from your followers to out-of-network. Be there to reply.
6. **Hashtags: 0–1.** Topic hashtags do nothing now; 3+ trigger a penalty. One *event* tag (e.g. #WorldCup2026) is the only justified use.
7. **Premium gives a real ~2–4× distribution multiplier.** If we're serious about the channel, the brand account being verified is worth it.

---

## 1. Two open-sources — know which one you're reading

| | **2023 "the-algorithm"** (legacy) | **Jan 2026 `xai-org/x-algorithm`** (current) |
|---|---|---|
| Architecture | Hand-engineered pipeline: candidate sourcing → **light ranker** → **Heavy Ranker** (neural net) → heuristics/filters | **Grok transformer ranker** that "eliminates every single hand-engineered feature and most heuristics" |
| Famous components | **SimClusters** (145k topic communities), **TweepCred** (PageRank reputation), **Real Graph** | Explicitly **drops SimClusters, TweepCred, content features** — the transformer learns relevance end-to-end |
| The numeric weights (reply 13.5, etc.) | **Hardcoded** in `home-mixer` scoring | **Not hardcoded** — the model *learns* these correlations from behaviour |

**Why this matters:** most "X algorithm weights" blog posts quote the **2023 hardcoded numbers**. Those are still the best public proxy for *what the algorithm rewards directionally*, because the 2026 Grok model was trained on the same human behaviour and learns the same patterns (replies signal quality, blocks signal harm). But treat the exact integers as **principles, not literal multipliers** in the current system.

---

## 2. The pipeline (2026 Grok version)

Per `xai-org/x-algorithm`, the For-You feed runs these stages per request:

1. **Query Hydration** — load user context (engagement history, following list).
2. **Candidate Sourcing** — two stores:
   - **Thunder (in-network):** posts from accounts you follow, in-memory, sub-ms lookups.
   - **Phoenix Retrieval (out-of-network):** ML similarity search across the global corpus.
   - Roughly **~50% in-network / ~50% out-of-network**, narrowed to **~1,500 candidates** from ~500M daily posts.
3. **Candidate Hydration** — enrich each post with metadata.
4. **Pre-Scoring Filters** — drop duplicates, too-old posts, self-posts, blocked authors, muted keywords, already-seen posts, ineligible subscriptions.
5. **Scoring**
   - **Phoenix Ranking:** a **Grok-based transformer** with *candidate isolation* — "candidates cannot attend to each other, only to the user context" (so scores are consistent + cacheable). It predicts **~15 engagement probabilities**.
   - **Weighted Scorer:** `Final Score = Σ (weight_i × P(action_i))` — positive actions get positive weights, negative actions negative weights.
6. **Selection** — sort by score, take top K.
7. **Post-Selection Filters** — visibility filtering (deleted / spam / violence / gore), conversation dedup.

**Mental model:** the model predicts *"how likely is this specific user to {reply / repost / like / block / …} this post?"*, multiplies each probability by that action's value (positive or negative), and sums. You win by maximising predicted *high-value* actions (replies, dwell, bookmarks) and minimising predicted *negative* ones.

---

## 3. The signal weights (legacy values = the directional truth)

These are the most-cited public weights (2023 production scorer; baseline **like = 0.5**). The current model isn't hardcoded to them but rewards the same ordering.

### Positive signals
| Signal | Weight | Note |
|---|---|---|
| **Reply that the author replies back to** | **+75** | The crown jewel. ≈150× a like. Real conversation. |
| Reply | **+13.5** | ~27× a like. |
| Profile click → then like/reply | +12 | They cared enough to visit you. |
| Conversation click → like/reply | +11 | Opened the thread and engaged. |
| Dwell ≥ 2 min on the post | +10 | "Stop the scroll." Long-read / good chart. |
| Bookmark | ~+10 | "I'll come back to this" = saved-worthy. |
| Repost / Retweet | +1.0 | Distribution, but lower than you'd think. |
| Like / Favorite | +0.5 | The baseline. |
| Video watch (≥50% / completes) | positive | Only if you post native video. |
| Follow author after seeing post | strong positive | The ultimate "this was great." |

> ⚠️ Discrepancy to know: some rundowns "normalise to like=1" and then quote "retweet ×20." That conflicts with the raw 1.0/0.5 = 2× ratio. Trust the **ordering** (reply-chain ≫ reply ≫ profile/convo/dwell/bookmark ≫ repost ≫ like), not any single multiplier.

### Negative signals (these dominate)
| Signal | Effect |
|---|---|
| **Report** | Major penalty; ~–15 to –35 to reputation (legacy). |
| **Block** | Strong penalty (legacy heavy-ranker ≈ –74 on the post; accrues to the account). |
| **Mute** | Strong negative. |
| **"Not interested" / "Show less often"** | Suppresses that content type for that user + signal to the model. |
| Unfollow after a post | Negative on the relationship. |

**One negative action can erase dozens of positive ones.** X explicitly optimises for *long-term retention* ("unregretted user-seconds"), not cheap engagement — so rage-bait/clickbait that earns a like-then-mute is a net loss.

---

## 4. Penalties & boosts

- **External links: ~30–50% reach reduction** (some report up to 50–90%; "near-zero median engagement for link posts on free accounts" since ~Mar 2025). The model learned that off-platform links correlate with session-end, so it lowers *all* predicted positive actions for link posts.
  - **Workaround (widely used):** put the link in the **first reply**, or in the post but not the first line; lead with the hook + native media.
- **Hashtags: 1–2 max.** 3+ ≈ "40% penalty" in rundowns; generic topic tags add nothing (the model reads semantics). Event tags only.
- **Premium / verification boost:** ~**4× in-network, ~2× out-of-network** visibility; rundowns claim ~**10× median impressions** vs free. Also unlocks longer posts, edit, reply-priority.
- **TweepCred (legacy):** 0–100 PageRank-style reputation; **threshold ≈ 65** — below it, supposedly only ~3 of your posts are considered for distribution at a time. Premium added points. *(Legacy concept; the Grok model claims to drop it, but account-quality/reputation effects clearly still exist in practice — new accounts have low reach until they build a track record.)*
- **Format claims (community data, not source):** native **text/images > video** on X specifically ("text ~30% more engagement than video"). Treat as directional; our chart *images* are the right native format.
- **Tone:** Grok does sentiment analysis; combative/low-quality tone can be suppressed regardless of raw engagement.
- **Posting cadence:** rundowns suggest **2–3 posts/day**, Tue–Thu mornings; consistency > bursts. Don't post 10× in an hour (self-competition + spam signals).

---

## 5. What this means for ChartSage (apply this)

Our account = a brand posting **data-showcase** content (per the social playbook). Mapping the algorithm to our loop:

1. **Link in the FIRST REPLY, not the main post.** This is the biggest change to our current plan. The showcase tweet should lead with the hook + the **chart image** (native), then drop the `chartsage.app/report/...` link as the first reply. (Tradeoff: the auto-unfurl card is nice, but the link penalty on the main post is real. Test both; lean link-in-reply for reach, link-in-post when the unfurl card *is* the hook.)
2. **Engineer for replies, then actually reply back.** End showcase posts with a light question ("Which stat surprised you?"). When people reply, **reply to them within the first hour** — that author-reply is the +75 signal and it compounds reach. This is the highest-ROI 20 minutes we can spend.
3. **Native chart images.** Attach the hero chart as an image (we already generate these). Don't rely on video. Don't rely solely on the unfurl.
4. **First 30–60 min = staffing.** Schedule posts for when you can be present to reply. A scheduled-and-abandoned post underperforms a posted-and-tended one.
5. **Bookmark-worthy = "save this."** Framing like "964 matches, one chart" / mini data-thread invites bookmarks (+10). A surprising, reference-able stat is bookmark fuel.
6. **One event hashtag at most** (#WorldCup2026), nothing generic. Skip hashtags entirely in **ads**.
7. **Don't be annoying.** No follow-bait, no rage-bait, no reply-spam in others' mentions (that earns blocks/mutes = nuclear). The playbook's Reddit-style discipline applies on X too.
8. **Consider Premium for the brand account.** A real 2–4× distribution multiplier is the single biggest free* lever (*it costs the sub) once we're posting consistently. Revisit after the first ~2 weeks of organic data.
9. **Threads for depth.** Post 1 = the hook + chart + (reply: link). Reply-thread = the secondary stats (the goals-decline, etc.). Self-reply threads keep dwell + conversation on-post.

### Our showcase post, algorithm-optimised
- **Main tweet:** hook + surprising stat + chart image + a soft question. *No link.*
- **Reply 1 (immediately, self):** "Full interactive report → chartsage.app/report/…?utm…"
- **Reply 2 (self, optional):** the bonus stat (goals 4.3→2.7) — keeps the thread alive.
- **Then:** reply to every genuine human reply for the first hour.

---

## 6. Caveats

- Exact weights are **legacy / community-reported**; the 2026 Grok model learns rather than hardcodes them. Use ordering + direction, not literal integers.
- Link-penalty magnitudes vary by source (30% to 90%); the *direction* (links hurt main-post reach) is consistent.
- Premium-boost multipliers are X-stated / community-measured, not from source code.
- The model retrains constantly. Re-check this doc before any expensive decision.

## Sources
- [xai-org/x-algorithm (GitHub, Jan 2026 open-source)](https://github.com/xai-org/x-algorithm)
- [TechCrunch — X open-sources its new algorithm (Jan 2026)](https://techcrunch.com/2026/01/20/x-open-sources-its-algorithm-while-facing-a-transparency-fine-and-grok-controversies/)
- [posteverywhere.ai — How the X/Twitter algorithm works in 2026 (source-code based)](https://posteverywhere.ai/blog/how-the-x-twitter-algorithm-works)
- [Social Media Today — X's ranking factors + weightings](https://www.socialmediatoday.com/news/x-formerly-twitter-open-source-algorithm-ranking-factors/759702/)
- [Sprout Social — How the Twitter/X algorithm works (2026)](https://sproutsocial.com/insights/twitter-algorithm/)
- [singhajit.com — X "For You" algorithm system design](https://singhajit.com/system-design/x-twitter-for-you-algorithm/)
- [ppc.land — How X silently kills your links](https://ppc.land/how-xs-algorithm-silently-kills-your-links-without-explicitly-penalizing-them/)
