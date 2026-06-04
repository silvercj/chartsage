# True-Private Reports — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-04
**Type:** Backend access-control + frontend private-state UX (privacy-critical)

## Goal
Make report visibility actually enforced. Today **any** report is viewable (and exportable) by **anyone with the link** — `GET /report/{id}` has no visibility/ownership check, and "Make private" only de-indexes (removes OG/sitemap) while the link still resolves the full report. A user who clicks "Make private" on **business data** can still open it logged-out in incognito. This fixes that: **reports are private by default**, viewable only by their owner, and become viewable-by-link only when the owner explicitly publishes.

## Decisions (from brainstorming)
- **Default = Private.** It's business data; nothing is exposed unless the owner publishes.
- **Two states, on the existing `is_public` boolean — NO schema change.** `is_public=false` (the current default) = **Private** (owner-only); `is_public=true` = **Public** (anyone-by-link + indexed). "Publish" → public; "Make private"/unpublish → private. The bug was that `is_public` was never enforced on read.
- **Ownership is identity-aware: authed `user_id` match OR anonymous `anon_id` match.** `apiFetch` already sends both `Authorization: Bearer` and `X-Anon-Id`, so an owner (signed-in *or* anonymous) is recognised on every call; nobody else is.
- **Anonymous users may publish** their own reports (gate on `anon_id`) — no account required to make a report public/shareable. (Extends today's authed-only publish.)
- **Non-owner of a private report gets `404`** (not 403) — don't reveal that a report exists at that id.

## Architecture

### 1. Access-control helper (the core) — `main.py`
Two small helpers, used by every report endpoint:

```python
def _is_owner(row: dict, identity: Identity) -> bool:
    if identity.user_id and row.get("user_id") == str(identity.user_id):
        return True
    if identity.anon_id and row.get("anon_id") == str(identity.anon_id):
        return True
    return False

def _resolve_report_access(db, identity, session_id, *, require_owner=False) -> tuple[dict, bool]:
    """Load a report + enforce visibility/ownership. Raises HTTPException.
    - missing            -> 404 NOT_FOUND
    - private & not owner -> 404 NOT_FOUND   (hide existence)
    - require_owner & not owner (public report) -> 403 NOT_OWNER
    Returns (row, is_owner)."""
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    is_owner = _is_owner(row, identity)
    if not row.get("is_public") and not is_owner:
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    if require_owner and not is_owner:
        raise HTTPException(403, detail={"code": "NOT_OWNER", "message": "You don't own this report."})
    return row, is_owner
```
This **replaces** the authed-only `_require_report_owner` (which 401'd anon callers and ignored `anon_id`). The new helper is anon-aware, so it also enables anon publish.

### 2. Per-endpoint rules
| Endpoint(s) | Identity dep | Rule |
|---|---|---|
| `GET /report/{id}` | `get_identity_optional` | view: `_resolve_report_access()` → return `report_json` |
| `GET /report/{id}/export.{pdf,pptx,xlsx,zip,md,html}` | `get_identity_optional` | view: `_resolve_report_access()` then render |
| `GET /report/{id}/meta` | `get_identity_optional` | never 404 for private; see §3 |
| `POST /report/{id}/publish`, `/unpublish` | `get_identity` | manage: `_resolve_report_access(require_owner=True)` (now anon-OK) |
| `POST /report/{id}/generate-more`, `/add-chart`, `/deepen`; `PATCH /report/{id}/layout` | `get_identity` | manage: `require_owner=True` |
| `GET /reports/public` | `get_db` | unchanged — lists `is_public=true` only |

*Implementation note:* audit each mutation/export endpoint's **current** identity handling and route it through the helper (several currently take `get_identity` or nothing and don't check ownership).

### 3. `/meta` must not leak private titles
`generateMetadata` (Next server component) calls `/meta` **server-side with no auth** → always resolves anon. So `/meta` must not return a private report's title/description to a non-owner:
- missing → 404.
- compute `owned` (anon-aware).
- **public** → full meta (`title`, `description`, `og_image_url`, `owned`, `is_public:true`).
- **private** → if `owned`, full meta; else **minimal**: `{is_public:false, owned:false, title:"Private report", description:"", og_image_url:null}`.

`generateMetadata` renders `robots: noindex` whenever not public.

### 4. Exports of private reports — the one non-trivial bit
PDF/PPTX exports render via **server-side Playwright** screenshotting the `/print` (and `/embed`/`/og`) Next routes, which fetch `GET /report/{id}` — but Playwright carries **no user token**, so for a private report that fetch would 404 and the export would fail for the legitimate owner. Options (plan picks one):
- **(A, recommended)** The export endpoint already authenticated the owner + has the `report_json` from `db`. Render from that data directly (pass it to the renderer / a server-only render mode) so the print route doesn't depend on the public `GET`.
- **(B)** Mint a short-lived, single-report signed token the server-side render passes back to `GET /report` (more moving parts).
The embed/`og` public routes are unaffected (they're only meaningful for public reports).

### 5. Frontend
- **`ReportClient`** (`apiFetch('/report/{id}')`): on `404`/`403`, render a clean **"This report is private or doesn't exist"** state instead of an error — with, for the anon case, a "Sign in if it's yours" link (an anon owner on a different device/browser has no `anon_id` match; signing in + claiming recovers access).
- **`/meta` `owned`** now true for anon owners too → the Toolbar's Share/Publish + "Make private" controls show for anon owners (so they can publish), per the anon-publish decision. The Toolbar fetches `/meta` **client-side** via `apiFetch` (so it sees the caller's identity); the server-side `generateMetadata` call always resolves anon and is used only for SEO meta, so `owned` is actionable only client-side.
- **`generateMetadata`**: `noindex` for non-public; public unchanged.
- Copy: the Share modal should state plainly that **publishing makes the report public and indexable by search engines**, and that **private is the default**.

### 6. Migration & rollout
- **No DB migration.** `is_public` already exists with the right default (`false`).
- **Behaviour change (the fix):** existing **unpublished** reports (`is_public=false`) become genuinely private — their bare links stop resolving for non-owners. Existing **published** reports (`is_public=true`) stay public. This is the intended correction; flag it in the deploy notes.

## Security considerations
- **`anon_id` is a bearer capability** (an unguessable client-stored UUID). Owner-gating by `anon_id` is acceptable for anon-owned reports; it is *not* a substitute for real auth, and is acknowledged as such (anon owners are nudged to sign in to persist/recover access).
- **404 over 403** for private non-owners → does not confirm a report exists at an id.
- The helper is **fail-closed**: any report whose `is_public` is false and isn't owned is hidden, across read **and** export paths.

## Scope & non-goals (v1)
**In:** the access helper + wiring it into read/export/manage endpoints; anon-aware ownership (incl. anon publish); `/meta` privacy; private-state frontend UX; export render path for private reports; Share-modal copy.
**Out (deferred):** a third "unlisted" state; per-recipient share links / invites; password-protected reports; audit logging of views; rate-limiting report reads.

## Verification
- **Backend (TDD, FakeDB):** owner (authed) views private → 200; non-owner → 404; **anon owner** (matching `anon_id`) views private → 200; different anon → 404; public report → 200 for anyone; publish by anon owner flips `is_public` → public; publish/mutation by non-owner → 404 (private) / 403 (public); every export endpoint enforces the same; `/meta` returns minimal for private non-owner, full for owner/public; `/reports/public` still lists only public.
- **Frontend:** private report shows the private-state UI for a non-owner; owner sees the report; Toolbar publish controls appear for anon owners; `generateMetadata` emits `noindex` for private.
- **Export:** an owner can export their **private** report (the render path doesn't 404).
- **Smoke (post-deploy):** generate a report (now private) → open its link in incognito → 404/private state; publish → link works + indexable; make private → link 404s again. (This is the exact bug repro.)
- Full `pytest` green; `tsc` clean; Vercel build Ready.
