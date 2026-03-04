import os

DEFAULT_RESPONSE = (
    "Entschuldigung, ich habe das nicht verstanden. "
    "Kannst du deine Frage bitte anders formulieren?"
)
MAX_MESSAGE_LENGTH = 500
GREETING_INTENT = "smalltalk_begruessung"
ERROR_THRESHOLD = 0.10
MIN_CONFIDENCE = 0.25
GREETING_CONFIDENCE_THRESHOLD = 0.55
MODEL_LOCK_TIMEOUT_SECONDS = float(os.environ.get("MODEL_LOCK_TIMEOUT_SECONDS", "2.0"))
MODEL_PREDICTION_TIMEOUT_SECONDS = float(
    os.environ.get("MODEL_PREDICTION_TIMEOUT_SECONDS", "2.5")
)

FOLLOW_UP_MARKERS = (
    "weiter",
    "weitermachen",
    "noch",
    "nochmal",
    "mehr",
    "tiefer",
    "genauer",
    "erklaer",
    "erklaeren",
    "beispiel",
    "vertief",
    "ausfuehr",
)

GENERIC_FOLLOW_UP_TOKENS = {
    "weiter",
    "weitermachen",
    "noch",
    "nochmal",
    "mehr",
    "tiefer",
    "genauer",
    "erklaer",
    "erklaeren",
    "beispiel",
    "vertief",
    "ausfuehrlich",
    "ausfuehren",
    "bitte",
    "mal",
}
