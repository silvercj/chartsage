#!/usr/bin/env bash
# Safe backend deploy for CODE-only changes: build the image and swap it on Cloud Run
# WITHOUT re-running the cloudbuild.yaml deploy step — whose substitution DEFAULTS are
# placeholders (SUPABASE_URL = https://YOUR_PROJECT.supabase.co, empty Stripe prices /
# UNLIMITED_ANON_IDS) that would clobber the live env. This swaps only the image, so every
# env var + secret is preserved.
#
#   scripts/deploy-backend.sh [<tag>]      # tag defaults to the short git sha
#
# For env/secret CHANGES, use the full config deploy instead, passing EVERY substitution:
#   gcloud builds submit --config cloudbuild.yaml \
#     --substitutions=_TAG=<sha>,_SUPABASE_URL=...,_STRIPE_PRICE_STARTER=...,\
#_STRIPE_PRICE_STANDARD=...,_STRIPE_PRICE_PRO=...,_UNLIMITED_ANON_IDS=...
set -euo pipefail

PROJECT=chartsage-497909
SERVICE=chartsage-backend
REGION=us-central1
TAG="${1:-$(git rev-parse --short HEAD)}"
IMAGE="gcr.io/${PROJECT}/${SERVICE}:${TAG}"

echo ">> building + pushing ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" .

echo ">> swapping ${SERVICE} -> ${IMAGE} (image only; env/secrets preserved)"
gcloud run services update "${SERVICE}" --image="${IMAGE}" --region="${REGION}"

URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')
echo ">> verifying ${URL}/health"
if curl -fsS "${URL}/health" >/dev/null; then
  echo "   OK — deployed ${IMAGE}"
else
  echo "   HEALTH CHECK FAILED"; exit 1
fi
