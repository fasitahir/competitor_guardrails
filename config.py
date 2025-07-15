from guardrails import Guard, OnFailAction
from validator import CheckCompetitorMentions  # <-- make sure this points to your class file
import os
from opentelemetry import metrics

# Disable metrics by setting a no-op meter provider
metrics.set_meter_provider(metrics.NoOpMeterProvider())
os.environ["OTEL_SDK_DISABLED"] = "true"

# Define competitor list (from your config)
competitors = [
    "Sastatickets",
    "Booking.com",
    "easytickets",
    "bookkaru",
    "checkin",
    "ticketwala",
    "ticket pak",
    "ticket price",
    "online ticket",
    "travelone",
    "pakwheels",
    "GoPakistanTravel"
]

# Guard setup
guard = Guard()
guard.name = "competitor-guard"

# Use custom NER-based validator
guard.use(
    CheckCompetitorMentions(
        competitors=competitors,
        on_fail=OnFailAction.EXCEPTION
    )
)
