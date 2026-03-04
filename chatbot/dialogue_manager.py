import json
import logging
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from textblob_de import TextBlobDE
from transitions import Machine

from .config import (
    DEFAULT_RESPONSE,
    MAX_MESSAGE_LENGTH,
    GREETING_INTENT,
    ERROR_THRESHOLD,
    MIN_CONFIDENCE,
    GREETING_CONFIDENCE_THRESHOLD,
    FOLLOW_UP_MARKERS,
    GENERIC_FOLLOW_UP_TOKENS,
)
from .nlp_utils import normalize_phrase, frage_bearbeitung, bow, get_ignore_words
from .model_engine import IntentClassifier

logger = logging.getLogger(__name__)


class DialogueManager:
    def __init__(self, base_dir: Path, classifier: IntentClassifier):
        self.base_dir = base_dir
        self.classifier = classifier
        self.ignore_words = get_ignore_words()

        self.dialogflow = {}
        self.intent_to_responses = {}
        self.intent_to_context = {}
        self.context_to_intents = {}
        self.phrase_to_intents = {}
        self.greeting_stems = set()
        self.synonym_index: List[Tuple[str, str, Set[str]]] = []

        self._load_dialogflow()

    def _load_dialogflow(self) -> None:
        chat_json_path = self.base_dir / "chat.json"
        try:
            with open(chat_json_path, encoding="utf8") as json_data:
                self.dialogflow = json.load(json_data)
        except Exception as e:
            logger.exception("Failed to load chat.json: %s", e)
            return

        for entry in self.dialogflow.get("dialogflow", []):
            intent = entry.get("intent")
            if not intent:
                continue

            self.intent_to_responses[intent] = entry.get("antwort", [])

            raw_context = (entry.get("kontext") or "").strip().lower()
            self.intent_to_context[intent] = raw_context
            self.context_to_intents.setdefault(raw_context, [])
            if intent not in self.context_to_intents[raw_context]:
                self.context_to_intents[raw_context].append(intent)

            for synonym in entry.get("synonym", []):
                key = normalize_phrase(synonym)
                if key:
                    self.phrase_to_intents.setdefault(key, [])
                    if intent not in self.phrase_to_intents[key]:
                        self.phrase_to_intents[key].append(intent)
                    self.synonym_index.append((intent, key, set(key.split())))

                # build greeting stems
                if intent == GREETING_INTENT:
                    self.greeting_stems.update(
                        frage_bearbeitung(synonym, self.ignore_words)
                    )

    def _get_exact_match_intent(self, frage: str) -> Optional[str]:
        normalized = normalize_phrase(frage)
        intents = self.phrase_to_intents.get(normalized, [])
        if not intents:
            return None
        for intent in intents:
            if intent != GREETING_INTENT:
                return intent
        return intents[0]

    def _looks_like_greeting(self, frage: str) -> bool:
        frage_stems = set(frage_bearbeitung(frage, self.ignore_words))
        return bool(frage_stems.intersection(self.greeting_stems))

    def _keyword_intent_match(self, frage: str) -> Optional[str]:
        normalized = normalize_phrase(frage)
        tokens = set(normalized.split())
        if not normalized:
            return None

        if any(
            phrase in normalized
            for phrase in (
                "ich will nicht mehr leben",
                "mir etwas antun",
                "ich kann nicht mehr",
                "suizid",
                "selbstmord",
            )
        ):
            return "mental_crisis_support"

        if tokens.intersection({"panik", "nervoes", "unruhig", "angst"}):
            return "mental_anxiety_support"
        if tokens.intersection(
            {"gestresst", "ueberfordert", "stress", "ausgebrannt", "burnout"}
        ):
            return "mental_stress_support"
        if tokens.intersection({"schlaf", "einschlafen", "durchschlafen", "muede"}):
            return "mental_sleep_support"
        if tokens.intersection(
            {"konzentration", "fokus", "abgelenkt", "prokrastination", "prokrastiniere"}
        ):
            return "mental_focus_support"
        if tokens.intersection(
            {"antriebslos", "energielos", "kraftlos", "erschoepft", "leer"}
        ):
            return "mental_energy_support"
        if tokens.intersection({"gruebeln", "grueble", "gedankenkarussell", "overthinke"}):
            return "mental_overthinking_support"
        if "body scan" in normalized or tokens.intersection({"koerperreise", "bodyscan"}):
            return "mental_body_scan"
        return None

    def _flexible_synonym_intent_match(self, frage: str) -> Optional[str]:
        normalized = normalize_phrase(frage)
        msg_tokens = set(normalized.split())
        if not msg_tokens:
            return None

        best_intent: Optional[str] = None
        best_score = 0.0

        for intent, synonym, synonym_tokens in self.synonym_index:
            if not synonym_tokens:
                continue

            overlap_tokens = msg_tokens.intersection(synonym_tokens)
            overlap = len(overlap_tokens) / len(synonym_tokens)
            coverage = len(overlap_tokens) / len(msg_tokens)
            token_score = (0.7 * overlap) + (0.3 * coverage)
            text_similarity = SequenceMatcher(None, normalized, synonym).ratio()
            score = max(token_score, text_similarity)

            if len(overlap_tokens) >= 2:
                score += 0.08

            if score > best_score:
                best_score = score
                best_intent = intent

        if not best_intent:
            return None

        if best_intent == GREETING_INTENT and not self._looks_like_greeting(frage):
            return None

        threshold = 0.78 if best_intent == GREETING_INTENT else 0.68
        if best_score >= threshold:
            return best_intent
        return None

    def _context_for_intent(self, intent: str) -> str:
        if not intent:
            return ""
        context = (self.intent_to_context.get(intent) or "").strip().lower()
        if context:
            return context
        if intent.startswith("mental_"):
            return "mental"
        if intent.startswith("smalltalk_"):
            return "smalltalk"
        return ""

    def _intents_for_context(self, context: str) -> List[str]:
        context = (context or "").strip().lower()
        intents = self.context_to_intents.get(context, [])
        if intents:
            return intents
        dynamic = []
        for intent in self.intent_to_responses:
            if self._context_for_intent(intent) == context:
                dynamic.append(intent)
        return dynamic

    def _is_follow_up_message(self, frage: str) -> bool:
        normalized = normalize_phrase(frage)
        if not normalized:
            return False
        return any(marker in normalized for marker in FOLLOW_UP_MARKERS)

    def _is_generic_follow_up_message(self, frage: str) -> bool:
        tokens = set(normalize_phrase(frage).split())
        if not tokens or len(tokens) > 5:
            return False
        return tokens.issubset(GENERIC_FOLLOW_UP_TOKENS)

    def _requested_context(self, frage: str) -> str:
        normalized = normalize_phrase(frage)
        tokens = set(normalized.split())
        if "check in" in normalized:
            return "mental"
        if any(
            word in tokens
            for word in {
                "mental",
                "check",
                "checkin",
                "check-in",
                "stress",
                "angst",
                "panik",
                "schlaf",
                "ueberfordert",
                "erschoepft",
                "atemuebung",
                "grounding",
                "krise",
                "konzentration",
                "fokus",
                "antriebslos",
                "gruebeln",
                "gedankenkarussell",
                "body",
                "scan",
            }
        ):
            return "mental"
        return ""

    def _is_meta_intent(self, intent: str) -> bool:
        if not intent:
            return False
        if (
            intent.startswith("feedback_")
            or intent.startswith("smalltalk_")
            or intent == "thema_auswahl"
        ):
            return True
        return self._context_for_intent(intent) in {
            "start",
            "ende",
            "navigation",
            "repair",
            "abschluss",
            "smalltalk",
        }

    def _pick_response(
        self, intent: str, previous_response: Optional[str] = None
    ) -> Optional[str]:
        responses = self.intent_to_responses.get(intent, [])
        if not responses:
            return None
        if previous_response and len(responses) > 1:
            alternatives = [r for r in responses if r != previous_response]
            if alternatives:
                return random.choice(alternatives)
        return random.choice(responses)

    def _pick_follow_up_intent(
        self, state: Dict[str, Any], preferred_context: str = ""
    ) -> str:
        last_intent = state.get("last_intent")
        last_context = (
            preferred_context
            or state.get("last_context")
            or self._context_for_intent(last_intent)
        )
        if not last_context and last_intent:
            return last_intent

        intents = self._intents_for_context(last_context)
        if not intents:
            return last_intent

        if last_intent in intents and len(intents) > 1 and last_context in {"mental"}:
            idx = intents.index(last_intent)
            return intents[(idx + 1) % len(intents)]

        if last_intent in intents:
            return last_intent
        return intents[0]

    def _klassifizieren(self, frage: str) -> List[Tuple[str, float]]:
        exact_match = self._get_exact_match_intent(frage)
        if exact_match:
            return [(exact_match, 1.0)]

        keyword_match = self._keyword_intent_match(frage)
        if keyword_match:
            return [(keyword_match, 0.99)]

        flexible_match = self._flexible_synonym_intent_match(frage)
        if flexible_match:
            return [(flexible_match, 0.95)]

        normalized = normalize_phrase(frage)
        tokens = set(normalized.split())
        if "check in" in normalized or "checkin" in tokens:
            return [("mental_checkin_start", 0.99)]

        bag_vector = bow(frage, self.classifier.words, self.ignore_words)
        ranked_results = self.classifier.classify_bag(bag_vector)

        if (
            ranked_results
            and ranked_results[0][0] == GREETING_INTENT
            and ranked_results[0][1] < GREETING_CONFIDENCE_THRESHOLD
            and not self._looks_like_greeting(frage)
        ):
            ranked_results = [
                item for item in ranked_results if item[0] != GREETING_INTENT
            ]

        filtered_results = [
            item
            for item in ranked_results
            if item[1] >= ERROR_THRESHOLD and item[1] >= MIN_CONFIDENCE
        ]
        return filtered_results

    def get_response(
        self, msg: str, state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        msg = msg.strip()
        if not msg:
            return {"text": ""}, None
        if len(msg) > MAX_MESSAGE_LENGTH:
            return {"text": "Bitte formuliere deine Nachricht etwas kürzer."}, None

        prior_response = state.get("last_response")
        follow_up = self._is_follow_up_message(msg)
        preferred_context = self._requested_context(msg)

        # Analyze Sentiment
        blob = TextBlobDE(msg)
        sentiment_polarity = blob.sentiment.polarity
        empathy_prefix = ""
        if sentiment_polarity < -0.5:
            empathy_prefix = "Das hoert sich wirklich anstrengend an. "
        elif sentiment_polarity > 0.5:
            empathy_prefix = "Das klingt doch schon mal positiv! "

        # Dynamic Placeholders: name extraction
        msg_lower = msg.lower()
        if "ich heisse " in msg_lower or "ich hei" in msg_lower:
            words = msg.split()
            # Very basic extraction: grab the last word
            if len(words) > 2:
                name = words[-1].strip(".,!?")
                state["user_name"] = name

        results = self._klassifizieren(msg)
        top_intent = results[0][0] if results else ""

        # Stateful Guided Exercise Logic
        active_exercise = state.get("active_exercise")
        buttons = []
        if active_exercise and follow_up:
            step = state.get("exercise_step", 0)
            if active_exercise == "mental_breathing_exercise":
                steps = [
                    "Schritt 1: Setz dich bequem hin und schliesse die Augen.",
                    "Schritt 2: Atme tief durch die Nase ein (4 Sekunden).",
                    "Schritt 3: Halte den Atem (4 Sekunden).",
                    "Schritt 4: Atme langsam durch den Mund aus (4 Sekunden).",
                ]
                if step < len(steps):
                    state["exercise_step"] = step + 1
                    buttons = [{"label": "Weiter", "value": "weiter"}]
                    return (
                        self._format_response(
                            steps[step], state, empathy_prefix, buttons
                        ),
                        active_exercise,
                    )
                else:
                    state["active_exercise"] = None
                    state["exercise_step"] = 0
                    return (
                        self._format_response(
                            "Gut gemacht! Wie fuehlst du dich jetzt?", state, "", []
                        ),
                        active_exercise,
                    )
            elif active_exercise == "mental_grounding":
                steps = [
                    "Schritt 1: Nenne 5 Dinge, die du im Raum sehen kannst.",
                    "Schritt 2: Beruehre 4 Dinge und spuere ihre Textur.",
                    "Schritt 3: Achte auf 3 verschiedene Geraeusche um dich herum.",
                    "Schritt 4: Kannst du 2 Dinge riechen?",
                    "Schritt 5: Wie schmeckt dein Mund gerade? Trink einen Schluck Wasser.",
                ]
                if step < len(steps):
                    state["exercise_step"] = step + 1
                    buttons = [{"label": "Weiter", "value": "weiter"}]
                    return (
                        self._format_response(
                            steps[step], state, empathy_prefix, buttons
                        ),
                        active_exercise,
                    )
                else:
                    state["active_exercise"] = None
                    state["exercise_step"] = 0
                    return (
                        self._format_response(
                            "Sehr gut. Hat dich das ein wenig geerdet?", state, "", []
                        ),
                        active_exercise,
                    )
            elif active_exercise == "mental_body_scan":
                steps = [
                    "Schritt 1: Setz oder leg dich bequem hin und schliesse, wenn moeglich, kurz die Augen.",
                    "Schritt 2: Richte die Aufmerksamkeit auf deine Stirn, Kiefer und Schultern. Loese dort bewusst Spannung.",
                    "Schritt 3: Wandere mit der Aufmerksamkeit langsam durch Brust, Bauch und Ruecken.",
                    "Schritt 4: Spuere Beine und Fuesse. Atme ruhig aus und lass Gewicht in den Boden sinken.",
                    "Schritt 5: Oeffne langsam die Augen und nenne ein Wort fuer dein aktuelles Koerpergefuehl.",
                ]
                if step < len(steps):
                    state["exercise_step"] = step + 1
                    buttons = [{"label": "Weiter", "value": "weiter"}]
                    return (
                        self._format_response(
                            steps[step], state, empathy_prefix, buttons
                        ),
                        active_exercise,
                    )
                state["active_exercise"] = None
                state["exercise_step"] = 0
                return (
                    self._format_response(
                        "Stark gemacht. Wenn du willst, machen wir als naechstes einen kurzen Check-in.",
                        state,
                        "",
                        [],
                    ),
                    active_exercise,
                )

        # When entering an exercise
        if top_intent in [
            "mental_breathing_exercise",
            "mental_grounding",
            "mental_body_scan",
        ]:
            state["active_exercise"] = top_intent
            state["exercise_step"] = 0

        if follow_up and state.get("last_intent"):
            follow_up_intent = self._pick_follow_up_intent(
                state,
                preferred_context=preferred_context or state.get("last_context"),
            )
            if follow_up_intent and (
                not top_intent
                or self._is_meta_intent(top_intent)
                or self._is_generic_follow_up_message(msg)
            ):
                response = self._pick_response(
                    follow_up_intent,
                    (
                        prior_response
                        if follow_up_intent == state.get("last_intent")
                        else None
                    ),
                )
                if response:
                    return (
                        self._format_response(response, state, empathy_prefix, []),
                        follow_up_intent,
                    )

        if preferred_context and (
            not results
            or (follow_up and (not top_intent or self._is_meta_intent(top_intent)))
        ):
            context_intents = self._intents_for_context(preferred_context)
            if context_intents:
                chosen_intent = self._pick_follow_up_intent(
                    {
                        "last_intent": state.get("last_intent"),
                        "last_context": preferred_context,
                    },
                    preferred_context=preferred_context,
                )
                response = self._pick_response(
                    chosen_intent,
                    (
                        prior_response
                        if chosen_intent == state.get("last_intent")
                        else None
                    ),
                )
                if response:
                    return (
                        self._format_response(response, state, empathy_prefix, []),
                        chosen_intent,
                    )

        if results:
            while results:
                intent = results[0][0]
                responses = self.intent_to_responses.get(intent, [])
                if responses:
                    response = self._pick_response(
                        intent,
                        prior_response if intent == state.get("last_intent") else None,
                    )
                    if response:
                        return (
                            self._format_response(response, state, empathy_prefix, []),
                            intent,
                        )
                results.pop(0)

        if follow_up:
            follow_up_intent = self._pick_follow_up_intent(
                state, preferred_context=preferred_context
            )
            if follow_up_intent:
                response = self._pick_response(
                    follow_up_intent,
                    (
                        prior_response
                        if follow_up_intent == state.get("last_intent")
                        else None
                    ),
                )
                if response:
                    return (
                        self._format_response(response, state, empathy_prefix, []),
                        follow_up_intent,
                    )

        fallback_buttons = [
            {"label": "Check-in", "value": "Ich brauche einen Check-in"},
            {"label": "Atemuebung", "value": "Atemuebung"},
            {"label": "Grounding", "value": "Grounding Uebung"},
        ]
        return self._format_response(DEFAULT_RESPONSE, state, "", fallback_buttons), None

    def _format_response(
        self,
        text: str,
        state: Dict[str, Any],
        empathy_prefix: str,
        buttons: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Helper to inject placeholders, prepend empathy, and pack into UI dict."""
        uname = state.get("user_name", "")
        # Remove empty bracket placeholders if we don't know the name yet.
        if uname:
            text = text.replace("[Name]", uname)
        else:
            text = text.replace(" [Name]", "").replace("[Name]", "")

        final_text = (empathy_prefix + text).strip()

        # Determine structured UI buttons
        # If no active exercise buttons, maybe suggest an exercise if it's stress/anxiety
        if (
            not buttons
            and state.get("last_context") == "mental"
            and not state.get("active_exercise")
        ):
            buttons = [
                {"label": "Atemübung", "value": "Atemuebung"},
                {"label": "Grounding", "value": "Grounding Uebung"},
                {"label": "Body-Scan", "value": "Body Scan"},
            ]

        return {"text": final_text, "buttons": buttons}
