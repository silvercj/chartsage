"""Lightweight alerting. Routes to Sentry when configured; always a safe no-op otherwise."""
import logging


def report_alert(message: str, **context) -> None:
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            for k, v in context.items():
                scope.set_extra(k, v)
            sentry_sdk.capture_message(message, level="warning")
    except Exception:
        logging.warning("[ALERT] %s %s", message, context or "")
