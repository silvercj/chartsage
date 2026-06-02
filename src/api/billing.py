# src/api/billing.py
"""Credit-pack catalogue for Stripe Checkout (SP4).

Credits live HERE (server-side source of truth); the price/amount lives in
Stripe (referenced by a Price ID from the environment). The client only ever
sends a package_id — never an amount — so a tampered client can neither change
what it pays nor how many credits it receives.
"""
import os

# package_id -> entry. credits = what we grant on purchase; price_env names the
# env var holding that pack's Stripe Price ID; price_display is the label shown
# on the pack card. The authoritative amount lives in Stripe (base currency USD;
# Adaptive Pricing, if enabled, localises it at checkout).
PACKAGES: dict[str, dict] = {
    "starter":  {"credits": 600,  "label": "Starter",  "price_env": "STRIPE_PRICE_STARTER",  "price_display": "$5"},
    "standard": {"credits": 2000, "label": "Standard", "price_env": "STRIPE_PRICE_STANDARD", "price_display": "$15"},
    "pro":      {"credits": 6000, "label": "Pro",      "price_env": "STRIPE_PRICE_PRO",      "price_display": "$40"},
}


def get_package(package_id: str) -> dict | None:
    """Catalogue entry for a package id, or None if unknown."""
    return PACKAGES.get(package_id)


def price_id_for(pkg: dict) -> str | None:
    """Resolve a package's Stripe Price ID from its env var (None if unset)."""
    return os.environ.get(pkg["price_env"]) or None


def public_catalogue() -> list[dict]:
    """The catalogue shaped for the frontend (no env/price internals)."""
    return [
        {"id": pid, "label": p["label"], "credits": p["credits"], "price_display": p["price_display"]}
        for pid, p in PACKAGES.items()
    ]
