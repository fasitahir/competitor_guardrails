import logging

# 🔇 Configure logging before any other imports that use OpenTelemetry
logging.basicConfig(
    filename='otel_debug.log',
    level=logging.WARNING,  # or ERROR if you want even fewer logs
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)

# Optional: silence OpenTelemetry-specific loggers
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.sdk").setLevel(logging.ERROR)

from guardrails import Guard, OnFailAction
from validator import CheckCompetitorMentions  # <-- make sure this points to your class file
import os
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

# Define competitor list (from your config)
competitors = [
    "Sastatickets",
    "Booking.com",
    "easytickets",
    "bookkaru",
    "ticketwala",
    "Ticketmaster"
    "ticket pak",
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
