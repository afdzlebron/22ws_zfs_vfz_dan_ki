import os
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, session

from .config import DEFAULT_RESPONSE
from .nlp_utils import ensure_nltk_data
from .model_engine import IntentClassifier
from .dialogue_manager import DialogueManager

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize dependencies
ensure_nltk_data(str(BASE_DIR / "nltk_data"))
classifier = IntentClassifier(BASE_DIR)
bot = DialogueManager(BASE_DIR, classifier)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "dev-secret-key-change-me"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": classifier.model_ready,
        }
    )


@app.route("/get", methods=["POST"])
def chatbot_response():
    if request.is_json:
        msg = (request.json or {}).get("msg", "")
    else:
        msg = request.form.get("msg", "")

    logger.info("Incoming message: %s", msg[:120])
    try:
        state = session.get("chat_state", {})
        res, intent = bot.get_response(msg, state)
        final_response = res or DEFAULT_RESPONSE

        if intent:
            state["last_intent"] = intent
            state["last_context"] = bot._context_for_intent(intent)
            state["last_response"] = final_response
            state["turns"] = min(int(state.get("turns", 0)) + 1, 1000)

        session["chat_state"] = state
        session.modified = True
    except Exception as exc:
        logger.exception("Failed to generate bot response: %s", exc)
        final_response = {"text": DEFAULT_RESPONSE, "buttons": []}

    logger.info("Outgoing response: %s", str(final_response)[:120])
    return jsonify(final_response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
