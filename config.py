from guardrails import Guard, OnFailAction
from validator import CheckCompetitorMentions  # <-- make sure this points to your class file
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

# Define competitor list (from your config)
competitors = [
    "Sastatickets",
    "easytickets",
    "bookkaru",
    "checkin",
    "ticketwala",
    "booking",
    "ticket pak",
    "ticket price",
    "online ticket",
    "travelone",
    "pakwheels",
    "pakistan railways",
    "GoPakistanTravel",
    "Pakistan Railways",
    "Pakistan Travels"
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
