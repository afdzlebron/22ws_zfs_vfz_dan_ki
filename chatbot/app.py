from flask import Flask, render_template, request, jsonify, Response, session

import json
import logging
import os
import pickle
import random
import re
import threading
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
import numpy as np
import tensorflow as tf
import tflearn

tf.compat.v1.disable_eager_execution()

stemmer = GermanStemmer()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.tflearn"
TRAINED_DATA_PATH = BASE_DIR / "trained_data"
CHAT_JSON_PATH = BASE_DIR / "chat.json"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

DEFAULT_RESPONSE = (
    "Entschuldigung, ich habe das nicht verstanden. "
    "Kannst du deine Frage bitte anders formulieren?"
)
MAX_MESSAGE_LENGTH = 500
GREETING_INTENT = "smalltalk_begruessung"
ERROR_THRESHOLD = 0.10
MIN_CONFIDENCE = 0.25
GREETING_CONFIDENCE_THRESHOLD = 0.55
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


def ensure_nltk_data():
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.append(str(NLTK_DATA_DIR))
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
    }
    for resource, name in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(name, download_dir=str(NLTK_DATA_DIR))
            except Exception as exc:
                logger.warning("Failed to download NLTK resource %s: %s", name, exc)

ensure_nltk_data()
try:
    STOPWORDS = set(stopwords.words("german"))
except LookupError as exc:
    logger.warning("NLTK stopwords not available yet: %s", exc)
    STOPWORDS = set()
IGNORE_WORDS = {"?", ".", ","}.union(STOPWORDS)

# wiederherstelle alle unsere Datenstrukturen
data = pickle.load(open(TRAINED_DATA_PATH, "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# importiere die Dialogdesign-Datei
with open(CHAT_JSON_PATH, encoding="utf8") as json_data:
    dialogflow = json.load(json_data)


def normalize_phrase(text):
    normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return " ".join(normalized.split())


INTENT_TO_RESPONSES = {
    entry.get("intent"): entry.get("antwort", [])
    for entry in dialogflow.get("dialogflow", [])
}
INTENT_TO_CONTEXT = {}
CONTEXT_TO_INTENTS = {}

PHRASE_TO_INTENTS = {}
for entry in dialogflow.get("dialogflow", []):
    intent = entry.get("intent")
    raw_context = (entry.get("kontext") or "").strip().lower()
    if intent:
        INTENT_TO_CONTEXT[intent] = raw_context
        CONTEXT_TO_INTENTS.setdefault(raw_context, [])
        if intent not in CONTEXT_TO_INTENTS[raw_context]:
            CONTEXT_TO_INTENTS[raw_context].append(intent)
    for synonym in entry.get("synonym", []):
        key = normalize_phrase(synonym)
        if not key:
            continue
        PHRASE_TO_INTENTS.setdefault(key, [])
        if intent not in PHRASE_TO_INTENTS[key]:
            PHRASE_TO_INTENTS[key].append(intent)

# Aufbau des neuronalen Netzes
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 88)
net = tflearn.fully_connected(net, 88)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Definiere das Modell und konfiguriere tensorboard
model = tflearn.DNN(net, tensorboard_dir=str(BASE_DIR / "train_logs"))
MODEL_READY = False



app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
# run_with_ngrok(app) 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL_READY,
            "nltk_data_dir": str(NLTK_DATA_DIR),
        }
    )

@app.route("/get", methods=["POST"])
def chatbot_response():
    if request.is_json:
        msg = (request.json or {}).get("msg", "")
    else:
        msg = request.form.get("msg", "")
    msg = msg.strip()
    if not msg:
        return ""
    if len(msg) > MAX_MESSAGE_LENGTH:
        return "Bitte formuliere deine Nachricht etwas kürzer."
    logger.info("Incoming message: %s", msg[:120])
    try:
        state = session.get("chat_state", {})
        res, intent = antwort(msg, state)
        final_response = res or DEFAULT_RESPONSE
        if intent:
            state["last_intent"] = intent
            state["last_context"] = context_for_intent(intent)
            state["last_response"] = final_response
            state["turns"] = min(int(state.get("turns", 0)) + 1, 1000)
            session["chat_state"] = state
    except Exception as exc:
        logger.exception("Failed to generate bot response: %s", exc)
        final_response = DEFAULT_RESPONSE
    logger.info("Outgoing response: %s", final_response[:120])
    return Response(final_response, mimetype="text/plain; charset=utf-8")


# chat functionalities
# Bearbeitung der Benutzereingaben, um einen bag-of-words zu erzeugen
def frageBearbeitung(frage):
    # tokenisiere die synonymen
    try:
        sentence_word = nltk.word_tokenize(frage, language="german")
    except LookupError as exc:
        logger.warning("NLTK punkt not available, falling back to split: %s", exc)
        sentence_word = frage.split()
    ######Korrektur Schreibfehler
    sentence_words = []
    for word in sentence_word:
        if word not in IGNORE_WORDS or word == 'weiter' or word == 'andere' or word == 'nicht':
            sentence_words.append(word)
    # stemme jedes Wort
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# Rückgabe bag of words array: 0 oder 1 für jedes Wort in der 'bag', die im Satz existiert
def bow(frage, words, show_details=False):
    sentence_words = frageBearbeitung(frage)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


# lade unsre gespeicherte Modell
try:
    model.load(str(MODEL_PATH))
    MODEL_READY = True
except Exception as exc:
    logger.exception("Failed to load model: %s", exc)

# Aufbau unseres Antwortprozessors.
# Erstellen einer Datenstruktur, die den Benutzerkontext enthält
context = {}
MODEL_LOCK = threading.Lock()


def get_exact_match_intent(frage):
    normalized = normalize_phrase(frage)
    intents = PHRASE_TO_INTENTS.get(normalized, [])
    if not intents:
        return None
    # Bei Mehrdeutigkeit (z. B. "Wer bist du") bevorzuge nicht den Begrüßungsintent.
    for intent in intents:
        if intent != GREETING_INTENT:
            return intent
    return intents[0]


def build_greeting_stems():
    stems = set()
    for entry in dialogflow.get("dialogflow", []):
        if entry.get("intent") != GREETING_INTENT:
            continue
        for synonym in entry.get("synonym", []):
            stems.update(frageBearbeitung(synonym))
    return stems


GREETING_STEMS = build_greeting_stems()


def looks_like_greeting(frage):
    frage_stems = set(frageBearbeitung(frage))
    return bool(frage_stems.intersection(GREETING_STEMS))


def context_for_intent(intent):
    if not intent:
        return ""
    context = (INTENT_TO_CONTEXT.get(intent) or "").strip().lower()
    if context:
        return context
    if intent.startswith("fun_"):
        return "fun"
    if intent.startswith("philosophie_"):
        return "philosophie"
    if intent.startswith("psychologie_"):
        return "psychologie"
    if intent.startswith("studentenleben_"):
        return "studium"
    if intent.startswith("smalltalk_"):
        return "smalltalk"
    return ""


def intents_for_context(context):
    context = (context or "").strip().lower()
    intents = CONTEXT_TO_INTENTS.get(context, [])
    if intents:
        return intents
    dynamic = []
    for intent in INTENT_TO_RESPONSES:
        if context_for_intent(intent) == context:
            dynamic.append(intent)
    return dynamic


def is_follow_up_message(frage):
    normalized = normalize_phrase(frage)
    if not normalized:
        return False
    return any(marker in normalized for marker in FOLLOW_UP_MARKERS)


def requested_context(frage):
    normalized = normalize_phrase(frage)
    tokens = set(normalized.split())
    if any(word in tokens for word in {"witz", "joke", "humor", "wortspiel", "kalauer"}):
        return "fun"
    if any(word in tokens for word in {"geschichte", "story", "scifi", "fantasy"}):
        return "story"
    if any(word in tokens for word in {"stoiker", "aristoteles", "nietzsche", "schopenhauer", "philosophie"}):
        return "philosophie"
    if any(word in tokens for word in {"kahneman", "stanovich", "psychologie", "bias", "prokrastination"}):
        return "psychologie"
    if any(word in tokens for word in {"studium", "studentenleben", "klausur", "ersti", "zeitmanagement"}):
        return "studium"
    return ""


def is_meta_intent(intent):
    if not intent:
        return False
    if intent.startswith("feedback_") or intent.startswith("smalltalk_") or intent == "thema_auswahl":
        return True
    return context_for_intent(intent) in {"start", "ende", "navigation", "repair", "abschluss", "smalltalk"}


def pick_response(intent, previous_response=None):
    responses = INTENT_TO_RESPONSES.get(intent, [])
    if not responses:
        return None
    if previous_response and len(responses) > 1:
        alternatives = [r for r in responses if r != previous_response]
        if alternatives:
            return random.choice(alternatives)
    return random.choice(responses)


def pick_follow_up_intent(state, preferred_context=""):
    last_intent = state.get("last_intent")
    last_context = preferred_context or state.get("last_context") or context_for_intent(last_intent)
    if not last_context and last_intent:
        return last_intent
    intents = intents_for_context(last_context)
    if not intents:
        return last_intent
    if last_intent in intents and len(intents) > 1 and last_context in {"fun", "story"}:
        idx = intents.index(last_intent)
        return intents[(idx + 1) % len(intents)]
    if last_intent in intents:
        return last_intent
    return intents[0]


def klassifizieren(frage):
    exact_match = get_exact_match_intent(frage)
    if exact_match:
        return [(exact_match, 1.0)]

    if not MODEL_READY:
        logger.error("Model is not ready yet.")
        return []
    bag_vector = bow(frage, words)
    if int(np.sum(bag_vector)) == 0:
        return []
    # generiere Wahrscheinlichkeiten von dem Modell
    try:
        # tflearn/tf1 inference is sensitive to threaded access under gunicorn.
        with MODEL_LOCK:
            results = model.predict([bag_vector])[0]
    except Exception as exc:
        logger.exception("Model prediction failed: %s", exc)
        return []
    ranked_results = sorted(
        [(classes[i], float(score)) for i, score in enumerate(results)],
        key=lambda item: item[1],
        reverse=True,
    )
    if (
        ranked_results
        and ranked_results[0][0] == GREETING_INTENT
        and ranked_results[0][1] < GREETING_CONFIDENCE_THRESHOLD
        and not looks_like_greeting(frage)
    ):
        ranked_results = [item for item in ranked_results if item[0] != GREETING_INTENT]

    filtered_results = [
        item
        for item in ranked_results
        if item[1] >= ERROR_THRESHOLD and item[1] >= MIN_CONFIDENCE
    ]
    return filtered_results


def antwort(frage, state=None):
    state = state or {}
    prior_response = state.get("last_response")
    follow_up = is_follow_up_message(frage)
    preferred_context = requested_context(frage)

    results = klassifizieren(frage)
    if follow_up and state.get("last_intent"):
        follow_up_intent = pick_follow_up_intent(
            state,
            preferred_context=preferred_context or state.get("last_context"),
        )
        top_intent = results[0][0] if results else ""
        if follow_up_intent and (not top_intent or is_meta_intent(top_intent)):
            response = pick_response(
                follow_up_intent,
                prior_response if follow_up_intent == state.get("last_intent") else None,
            )
            if response:
                return response, follow_up_intent

    if preferred_context and (follow_up or not results):
        context_intents = intents_for_context(preferred_context)
        if context_intents:
            chosen_intent = pick_follow_up_intent(
                {
                    "last_intent": state.get("last_intent"),
                    "last_context": preferred_context,
                },
                preferred_context=preferred_context,
            )
            response = pick_response(chosen_intent, prior_response if chosen_intent == state.get("last_intent") else None)
            if response:
                return response, chosen_intent

    # Wenn wir eine Klassifizierung haben, dann suchen wir das passende dialog-intent
    if results:
        # loop solange es Übereinstimmungen gibt, die verarbeitet werden sollen
        while results:
            intent = results[0][0]
            responses = INTENT_TO_RESPONSES.get(intent, [])
            if responses:
                response = pick_response(
                    intent,
                    prior_response if intent == state.get("last_intent") else None,
                )
                if response:
                    return response, intent
            results.pop(0)

    if follow_up:
        follow_up_intent = pick_follow_up_intent(state, preferred_context=preferred_context)
        if follow_up_intent:
            response = pick_response(
                follow_up_intent,
                prior_response if follow_up_intent == state.get("last_intent") else None,
            )
            if response:
                return response, follow_up_intent

    return DEFAULT_RESPONSE, None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
