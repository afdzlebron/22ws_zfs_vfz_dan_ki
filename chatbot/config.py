import os

DEFAULT_RESPONSE = (
    "Danke fuer deine Nachricht. Ich bin fuer dich da, habe dich aber noch nicht klar verstanden. "
    "Magst du kurz sagen, ob es um Stress, Angst, Schlaf oder eine Uebung geht?"
)
MAX_MESSAGE_LENGTH = 500
GREETING_INTENT = "smalltalk_begruessung"
ERROR_THRESHOLD = 0.10
MIN_CONFIDENCE = 0.25
GREETING_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_CLARIFY_THRESHOLD = 0.52
LOW_CONFIDENCE_MARGIN_THRESHOLD = 0.12
MODEL_LOCK_TIMEOUT_SECONDS = float(os.environ.get("MODEL_LOCK_TIMEOUT_SECONDS", "2.0"))
MODEL_PREDICTION_TIMEOUT_SECONDS = float(
    os.environ.get("MODEL_PREDICTION_TIMEOUT_SECONDS", "2.5")
)
DEFAULT_RESPONSE_MODE = "normal"

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
