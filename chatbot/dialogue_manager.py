import json
import logging
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from textblob_de import TextBlobDE
except Exception:  # pragma: no cover - environment dependent import
    TextBlobDE = None

from .config import (
    DEFAULT_RESPONSE,
    MAX_MESSAGE_LENGTH,
    GREETING_INTENT,
    ERROR_THRESHOLD,
    MIN_CONFIDENCE,
    GREETING_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_CLARIFY_THRESHOLD,
    LOW_CONFIDENCE_MARGIN_THRESHOLD,
    DEFAULT_RESPONSE_MODE,
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
        self.response_components = {}
        self.intent_to_components: Dict[str, Dict[str, Any]] = {}
        self.intent_labels = {
            "mental_stress_support": "Stress",
            "mental_anxiety_support": "Angst",
            "mental_sleep_support": "Schlaf",
            "mental_focus_support": "Fokus",
            "mental_energy_support": "Antrieb",
            "mental_overthinking_support": "Gedankenkreisen",
            "mental_breathing_exercise": "Atem",
            "mental_grounding": "Grounding",
            "mental_body_scan": "Koerperwahrnehmung",
            "mental_crisis_support": "akute Unterstuetzung",
        }
        self.follow_up_plans = {
            "mental_checkin_start": [
                "mental_stress_support",
                "mental_anxiety_support",
                "mental_sleep_support",
            ],
            "mental_stress_support": [
                "mental_breathing_exercise",
                "mental_grounding",
                "mental_focus_support",
            ],
            "mental_anxiety_support": [
                "mental_breathing_exercise",
                "mental_grounding",
                "mental_overthinking_support",
            ],
            "mental_sleep_support": [
                "mental_body_scan",
                "mental_breathing_exercise",
                "mental_overthinking_support",
            ],
            "mental_focus_support": [
                "mental_breathing_exercise",
                "mental_energy_support",
                "mental_grounding",
            ],
            "mental_energy_support": [
                "mental_body_scan",
                "mental_focus_support",
                "mental_breathing_exercise",
            ],
            "mental_overthinking_support": [
                "mental_grounding",
                "mental_breathing_exercise",
                "mental_body_scan",
            ],
        }
        self.style_aliases = {
            "warm": {"warm", "freundlich", "empathisch", "menschlich"},
            "direct": {"direkt", "sachlich", "klar"},
            "brief": {"kurz", "knapp", "kompakt"},
            "detailed": {"detailliert", "ausfuehrlich", "tiefer"},
        }

        self._load_dialogflow()

    def _load_dialogflow(self) -> None:
        chat_json_path = self.base_dir / "chat.json"
        try:
            with open(chat_json_path, encoding="utf8") as json_data:
                self.dialogflow = json.load(json_data)
        except Exception as e:
            logger.exception("Failed to load chat.json: %s", e)
            return

        self.response_components = self.dialogflow.get("response_components", {})
        for entry in self.dialogflow.get("dialogflow", []):
            intent = entry.get("intent")
            if not intent:
                continue

            self.intent_to_responses[intent] = entry.get("antwort", [])
            # Also store components per intent if present in the dialogflow entry
            components = entry.get("components")
            if isinstance(components, dict):
                self.intent_to_components[intent] = components

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

    def _default_quick_buttons(self) -> List[Dict[str, str]]:
        return [
            {"label": "Check-in", "value": "Ich brauche einen Check-in"},
            {"label": "Stress", "value": "Ich bin gestresst"},
            {"label": "Schlaf", "value": "Ich kann nicht schlafen"},
            {"label": "Atemuebung", "value": "Atemuebung"},
            {"label": "Hilfe", "value": "/hilfe"},
        ]

    def _is_mental_intent(self, intent: str) -> bool:
        return bool(intent) and intent.startswith("mental_")

    def _topic_label(self, intent: str) -> str:
        return self.intent_labels.get(intent, "dein Thema")

    def _dedupe_buttons(self, buttons: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        unique = []
        for button in buttons:
            value = button.get("value", "")
            if value in seen:
                continue
            seen.add(value)
            unique.append(button)
        return unique

    def _pick_non_repeating_option(
        self,
        options: List[str],
        state: Dict[str, Any],
        history_key: str,
        window: int = 4,
    ) -> str:
        options = [
            item.strip() for item in options if isinstance(item, str) and item.strip()
        ]
        if not options:
            return ""

        history = state.setdefault(history_key, [])
        if not isinstance(history, list):
            history = []
            state[history_key] = history

        recent = set(history[-window:])
        candidates = [item for item in options if item not in recent]
        choice = random.choice(candidates or options)

        history.append(choice)
        if len(history) > 20:
            del history[:-20]
        return choice

    def _extract_style_preference(self, frage: str) -> Optional[str]:
        normalized = normalize_phrase(frage)
        if not normalized:
            return None

        tokens = set(normalized.split())
        for style, aliases in self.style_aliases.items():
            if normalized.startswith("/stil ") and aliases.intersection(tokens):
                return style
            if "sprechstil" in normalized and aliases.intersection(tokens):
                return style
            if "antworte" in normalized and aliases.intersection(tokens):
                return style
            if "sei" in normalized and aliases.intersection(tokens):
                return style
        return None

    def _style_confirmation(self, style: str) -> str:
        messages = {
            "warm": "Alles klar, ich antworte jetzt waermer und empathischer.",
            "direct": "Alles klar, ich antworte jetzt direkter und klarer.",
            "brief": "Alles klar, ich halte mich jetzt kurz und fokussiert.",
            "detailed": "Alles klar, ich antworte jetzt ausfuehrlicher und strukturierter.",
        }
        return messages.get(style, messages["warm"])

    def _components_for_intent(self, intent: str) -> Dict[str, Any]:
        components = self.intent_to_components.get(intent, {})
        return components if isinstance(components, dict) else {}

    def _next_question_kind(self, state: Dict[str, Any]) -> str:
        previous_kind = state.get("last_question_kind", "closed")
        next_kind = "open" if previous_kind == "closed" else "closed"
        state["last_question_kind"] = next_kind
        return next_kind

    def _strip_question_marks(self, text: str) -> str:
        return re.sub(r"\s*\?+\s*", ". ", text).strip()

    def _style_validation_defaults(self, style: str) -> List[str]:
        defaults = self.response_components.get("defaults", {}).get("validation", {})
        options = defaults.get(style, [])
        if not options:
            return ["Verstanden.", "Okay, ich hab dich."]
        return options

    def _build_reflection(self, intent: str, state: Dict[str, Any], msg: str) -> str:
        """Light mirroring using extracted slots or short snippets."""
        # Slot-based rephrasing
        if intent == "mental_sleep_support" and isinstance(
            state.get("sleep_hours"), (int, float)
        ):
            return f"Du sagtest, dass dein Schlaf gerade bei etwa {state['sleep_hours']}h liegt."

        if intent == "mental_stress_support" and isinstance(
            state.get("stress_level"), int
        ):
            return f"Du hast deinen Stresslevel bei {state['stress_level']}/10 eingeordnet."

        # Snippet-based rephrasing
        cleaned = " ".join(msg.strip().split())
        if len(cleaned) >= 15:
            # Simple rephrasing logic: "You said X..."
            snippet = cleaned[:80].rstrip(".,!?")
            # Try to grab a more meaningful part if possible (after "weil", "dass", etc.)
            match = re.search(r"\b(weil|dass|da|wenn)\b\s+(.*)", cleaned, re.IGNORECASE)
            if match:
                snippet = match.group(2)[:60].rstrip(".,!?")

            templates = [
                "Du meintest: '{snippet}'.",
                "Ich habe verstanden, dass '{snippet}' dich gerade beschaeftigt.",
                "Du hast erwaehnt: '{snippet}'.",
            ]
            template = random.choice(templates)
            return template.format(snippet=snippet)

        return ""

    def _question_options_for_intent(
        self, intent: str, question_kind: str
    ) -> List[str]:
        components = self._components_for_intent(intent)
        questions = components.get("question", {})
        if isinstance(questions, dict):
            options = questions.get(question_kind, [])
            if isinstance(options, list):
                clean = [
                    item for item in options if isinstance(item, str) and item.strip()
                ]
                if clean:
                    return clean

        fallback = self.response_components.get(intent, {}).get("question", {})
        options = fallback.get(question_kind, [])
        return [item for item in options if isinstance(item, str) and item.strip()]

    def _compose_planned_response(
        self,
        intent: str,
        state: Dict[str, Any],
        base_response: str,
        msg: str,
        follow_up: bool,
    ) -> str:
        if not self._is_mental_intent(intent) or intent == "mental_crisis_support":
            return base_response

        style = state.get("conversation_style", "warm")
        components = self._components_for_intent(intent)

        # 1. Acknowledge (Validation)
        validation_options = components.get("validation", [])
        if not isinstance(validation_options, list) or not validation_options:
            validation_options = self._style_validation_defaults(style)

        validation = self._pick_non_repeating_option(
            validation_options,
            state,
            "recent_validation_phrases",
            window=6,
        )

        # 2. Reflect (Light Mirroring)
        reflection = ""
        if style not in {"brief", "direct"}:
            reflection = self._build_reflection(intent, state, msg)

        # 3. Action (One concrete step)
        action_text = base_response
        if not action_text:
            action_options = components.get("action", [])
            if isinstance(action_options, list):
                action_text = self._pick_non_repeating_option(
                    action_options,
                    state,
                    "recent_action_phrases",
                    window=6,
                )

        if not action_text:
            return base_response

        # Strong anti-repetition: block same transition/opening patterns
        if action_text.strip() == state.get("last_action_text", "").strip():
            action_options = components.get("action", [])
            if isinstance(action_options, list) and len(action_options) > 1:
                action_text = random.choice(
                    [a for a in action_options if a.strip() != action_text.strip()]
                )

        state["last_action_text"] = action_text

        # Enforce "one question per turn": strip questions from action part
        if "?" in action_text:
            action_text = self._strip_question_marks(action_text)

        # 4. Better follow-up strategy: alternate open vs closed question
        question_text = ""
        question_kind = self._next_question_kind(state)

        # If it's a follow-up turn (the user said things like "more", "next"),
        # we might want to guide them more directly with a closed question?
        # Or just stick to the alternation. Let's stick to alternation for variety.

        question_options = self._question_options_for_intent(intent, question_kind)
        if question_options:
            question_text = self._pick_non_repeating_option(
                question_options,
                state,
                "recent_question_phrases",
                window=6,
            )
            question_text = question_text.strip().rstrip(".! ")
            if not question_text.endswith("?"):
                question_text += "?"

        parts = [validation]
        if reflection:
            parts.append(reflection)
        parts.append(action_text.strip())
        if question_text:
            parts.append(question_text)

        planned = " ".join(part for part in parts if part).strip()
        return planned

    def _extract_response_mode(self, frage: str) -> Optional[str]:
        normalized = normalize_phrase(frage)
        if not normalized:
            return None

        short_markers = {"kurz", "knapp", "kompakt"}
        normal_markers = {"normal", "standard"}
        detailed_markers = {"ausfuehrlich", "detailliert", "lang"}

        if normalized.startswith("modus "):
            mode_value = normalized.split()[-1]
            if mode_value in short_markers:
                return "short"
            if mode_value in normal_markers:
                return "normal"
            if mode_value in detailed_markers:
                return "detailed"

        if normalized.startswith("/modus "):
            mode_value = normalized.split()[-1]
            if mode_value in short_markers:
                return "short"
            if mode_value in normal_markers:
                return "normal"
            if mode_value in detailed_markers:
                return "detailed"

        if "antwortmodus" in normalized or "antworten" in normalized:
            tokens = set(normalized.split())
            if tokens.intersection(short_markers):
                return "short"
            if tokens.intersection(normal_markers):
                return "normal"
            if tokens.intersection(detailed_markers):
                return "detailed"
        return None

    def _is_help_request(self, frage: str) -> bool:
        normalized = normalize_phrase(frage)
        if not normalized:
            return False
        if normalized in {"hilfe", "/hilfe", "help", "/help"}:
            return True
        return "was kannst du" in normalized or "hilfe" in normalized

    def _extract_slots(self, frage: str, state: Dict[str, Any]) -> None:
        normalized = normalize_phrase(frage)
        if not normalized:
            return

        stress_match = re.search(
            r"(?:stress(?:level)?\s*)?(10|[1-9])(?:\s*(?:von|/)?\s*10)?",
            normalized,
        )
        if stress_match:
            level = int(stress_match.group(1))
            state["stress_level"] = level

        sleep_match = re.search(
            r"\b(\d{1,2})(?:[\.,](\d))?\s*(?:h|std|stunden)\b",
            normalized,
        )
        if sleep_match:
            hours = float(sleep_match.group(1))
            decimal_part = sleep_match.group(2)
            if decimal_part:
                hours += float(f"0.{decimal_part}")
            state["sleep_hours"] = round(hours, 1)

    def _is_plain_stress_rating(self, frage: str) -> bool:
        normalized = normalize_phrase(frage)
        if not normalized:
            return False
        return bool(re.fullmatch(r"(10|[1-9])(?:\s*(?:von|/)?\s*10)?", normalized))

    def _should_clarify_intent(
        self, results: List[Tuple[str, float]], follow_up: bool
    ) -> bool:
        if follow_up or not results:
            return False
        top_score = float(results[0][1])
        if top_score >= LOW_CONFIDENCE_CLARIFY_THRESHOLD:
            return False
        if len(results) == 1:
            return True
        margin = top_score - float(results[1][1])
        return margin <= LOW_CONFIDENCE_MARGIN_THRESHOLD

    def _clarification_buttons(
        self, results: List[Tuple[str, float]]
    ) -> List[Dict[str, str]]:
        """Richer repair flow: show 2-3 specific options instead of generic fallback."""
        suggestions = {
            "mental_stress_support": {"label": "Stress", "value": "Stress bewaeltigen"},
            "mental_anxiety_support": {"label": "Angst", "value": "Umgang mit Angst"},
            "mental_sleep_support": {"label": "Schlaf", "value": "Schlaf verbessern"},
            "mental_focus_support": {"label": "Fokus", "value": "Besser konzentrieren"},
            "mental_energy_support": {
                "label": "Antrieb",
                "value": "Mehr Energie finden",
            },
            "mental_overthinking_support": {
                "label": "Gedankenstopp",
                "value": "Gedankenkreisen stoppen",
            },
            "mental_breathing_exercise": {
                "label": "Atemuebung",
                "value": "Atemuebung starten",
            },
            "mental_grounding": {"label": "Grounding", "value": "Grounding Uebung"},
            "mental_body_scan": {"label": "Body Scan", "value": "Body Scan starten"},
        }

        buttons = []
        # Take up to 3 high-confidence intents for specific 'Did you mean...?' options
        for intent, score in results[:3]:
            if intent in suggestions:
                buttons.append(suggestions[intent])

        if len(buttons) < 2:
            buttons = self._default_quick_buttons()[:3]

        return self._dedupe_buttons(buttons)

    def _clarification_text(self, results: List[Tuple[str, float]]) -> str:
        """Richer repair text for low confidence."""
        top_labels = [
            self._topic_label(intent)
            for intent, _score in results[:3]
            if self._is_mental_intent(intent)
        ]
        top_labels = [label for label in top_labels if label]

        if len(top_labels) >= 2:
            joined = " oder ".join(top_labels[:3])
            return (
                f"Ich bin mir noch nicht ganz sicher. Meinst du eher {joined}? "
                "Waehle einfach die passende Richtung:"
            )

        return "Ich habe dich noch nicht klar verstanden. Meinst du eine dieser Richtungen?"

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
        if tokens.intersection(
            {"gruebeln", "grueble", "gedankenkarussell", "overthinke"}
        ):
            return "mental_overthinking_support"
        if "body scan" in normalized or tokens.intersection(
            {"koerperreise", "bodyscan"}
        ):
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
        self,
        intent: str,
        previous_response: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        responses = self.intent_to_responses.get(intent, [])
        if not responses:
            return None

        candidates = list(responses)
        if previous_response and len(candidates) > 1:
            no_prev = [r for r in candidates if r != previous_response]
            if no_prev:
                candidates = no_prev

        if state is not None and len(candidates) > 1:
            recent = state.setdefault("recent_raw_responses", [])
            if not isinstance(recent, list):
                recent = []
                state["recent_raw_responses"] = recent
            recent_set = set(recent[-4:])
            non_repeating = [r for r in candidates if r not in recent_set]
            if non_repeating:
                candidates = non_repeating

        choice = random.choice(candidates)
        if state is not None:
            recent = state.setdefault("recent_raw_responses", [])
            if isinstance(recent, list):
                recent.append(choice)
                if len(recent) > 20:
                    del recent[:-20]
        return choice

    def _pick_follow_up_intent(
        self,
        state: Dict[str, Any],
        preferred_context: str = "",
        rotate_for_progression: bool = False,
    ) -> str:
        last_intent = state.get("last_intent")
        last_context = (
            preferred_context
            or state.get("last_context")
            or self._context_for_intent(last_intent)
        )
        if last_intent and not rotate_for_progression:
            return last_intent

        if last_intent in self.follow_up_plans:
            plan = [
                candidate
                for candidate in self.follow_up_plans[last_intent]
                if candidate in self.intent_to_responses
            ]
            if plan:
                idx = int(state.get("follow_up_plan_index", 0))
                state["follow_up_plan_index"] = idx + 1
                return plan[idx % len(plan)]

        if not last_context and last_intent:
            return last_intent

        intents = self._intents_for_context(last_context)
        if not intents:
            return last_intent

        if (
            rotate_for_progression
            and last_intent in intents
            and len(intents) > 1
            and last_context in {"mental"}
        ):
            idx = intents.index(last_intent)
            return intents[(idx + 1) % len(intents)]

        if last_intent in intents:
            return last_intent
        return intents[0]

    def _choose_primary_intent(
        self,
        results: List[Tuple[str, float]],
        state: Dict[str, Any],
        follow_up: bool,
        msg: str,
    ) -> str:
        if not results:
            return ""

        top_intent, top_score = results[0]
        last_intent = state.get("last_intent", "")

        if follow_up and last_intent:
            if self._is_generic_follow_up_message(msg):
                return last_intent
            if (
                self._context_for_intent(last_intent)
                == self._context_for_intent(top_intent)
                and top_intent != last_intent
                and top_score < 0.75
            ):
                return last_intent

        topic_intent = state.get("topic_intent", "")
        if (
            topic_intent
            and top_intent != topic_intent
            and self._context_for_intent(top_intent)
            == self._context_for_intent(topic_intent)
            and top_score < 0.55
        ):
            return topic_intent

        return top_intent

    def _build_intent_response(
        self,
        intent: str,
        state: Dict[str, Any],
        response: str,
        empathy_prefix: str,
        buttons: List[Dict[str, str]],
        msg: str = "",
        follow_up: bool = False,
        use_planner: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        state["fallback_count"] = 0
        if self._is_mental_intent(intent):
            state["topic_intent"] = intent

        final_response = response
        if use_planner:
            final_response = self._compose_planned_response(
                intent=intent,
                state=state,
                base_response=response,
                msg=msg,
                follow_up=follow_up,
            )

        return (
            self._format_response(
                final_response,
                state,
                empathy_prefix,
                buttons,
                intent=intent,
            ),
            intent,
        )

    def _fallback_response(
        self, state: Dict[str, Any], preferred_context: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        state["fallback_count"] = int(state.get("fallback_count", 0)) + 1
        fallback_count = state["fallback_count"]
        candidate_intents = [
            intent
            for intent in state.get("last_candidate_intents", [])
            if isinstance(intent, str)
        ]
        candidate_labels = [
            self._topic_label(intent)
            for intent in candidate_intents
            if self._is_mental_intent(intent)
        ]
        candidate_labels = [label for label in candidate_labels if label]

        if fallback_count >= 2:
            last_topic = state.get("topic_intent") or state.get("last_intent", "")
            if self._is_mental_intent(last_topic):
                repair_text = (
                    f"Ich bin noch nicht ganz sicher, was du brauchst. "
                    f"Sollen wir bei {self._topic_label(last_topic)} bleiben oder kurz wechseln?"
                )
            elif preferred_context == "mental" or state.get("last_context") == "mental":
                repair_text = (
                    "Ich verstehe dich noch nicht eindeutig. "
                    "Waehle bitte kurz das Thema, dann mache ich direkt passend weiter."
                )
            else:
                repair_text = (
                    "Ich habe dich noch nicht klar verstanden. "
                    "Nimm gern eine der schnellen Optionen, dann antworte ich gezielter."
                )

            if candidate_labels:
                repair_text += (
                    " Soll ich eher bei "
                    + ", ".join(candidate_labels[:3])
                    + " einsteigen?"
                )

            candidate_buttons = self._clarification_buttons(
                [(intent, 0.0) for intent in candidate_intents]
            )
            return (
                self._format_response(
                    repair_text,
                    state,
                    "",
                    candidate_buttons + self._default_quick_buttons(),
                    intent="thema_auswahl",
                ),
                None,
            )

        return (
            self._format_response(
                DEFAULT_RESPONSE,
                state,
                "",
                self._default_quick_buttons(),
                intent="thema_auswahl",
            ),
            None,
        )

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
        if isinstance(prior_response, dict):
            prior_response = prior_response.get("text")
        state.setdefault("response_mode", DEFAULT_RESPONSE_MODE)
        state.setdefault("conversation_style", "warm")

        requested_mode = self._extract_response_mode(msg)
        if requested_mode:
            state["response_mode"] = requested_mode
            mode_text = {
                "short": "Alles klar, ich antworte ab jetzt kuerzer und direkter.",
                "normal": "Alles klar, ich antworte wieder im normalen Modus.",
                "detailed": "Alles klar, ich antworte ab jetzt ausfuehrlicher.",
            }
            return self._build_intent_response(
                "thema_auswahl",
                state,
                mode_text.get(requested_mode, mode_text["normal"]),
                "",
                [],
                msg=msg,
                use_planner=False,
            )

        requested_style = self._extract_style_preference(msg)
        if requested_style:
            state["conversation_style"] = requested_style
            return self._build_intent_response(
                "thema_auswahl",
                state,
                self._style_confirmation(requested_style),
                "",
                [],
                msg=msg,
                use_planner=False,
            )

        if self._is_help_request(msg):
            help_response = self._pick_response("thema_auswahl", state=state) or (
                "Ich kann dir bei Stress, Angst, Schlaf, Fokus, Antrieb und "
                "Entspannungsuebungen helfen."
            )
            return self._build_intent_response(
                "thema_auswahl",
                state,
                help_response,
                "",
                [],
                msg=msg,
                use_planner=False,
            )

        self._extract_slots(msg, state)
        follow_up = self._is_follow_up_message(msg)
        generic_follow_up = follow_up and self._is_generic_follow_up_message(msg)
        if follow_up:
            if generic_follow_up:
                state["generic_follow_up_count"] = (
                    int(state.get("generic_follow_up_count", 0)) + 1
                )
            else:
                state["generic_follow_up_count"] = 0
        else:
            state["generic_follow_up_count"] = 0
        preferred_context = self._requested_context(msg)

        # Analyze Sentiment
        empathy_prefix = ""
        if len(msg.split()) >= 4 and TextBlobDE is not None:
            try:
                blob = TextBlobDE(msg)
                sentiment_polarity = blob.sentiment.polarity
                if sentiment_polarity < -0.5:
                    empathy_prefix = "Das hoert sich wirklich anstrengend an. "
                elif sentiment_polarity > 0.5:
                    empathy_prefix = "Das klingt doch schon mal positiv! "
            except Exception:
                logger.debug("Sentiment analysis skipped for input.")

        # Dynamic Placeholders: name extraction
        msg_lower = msg.lower()
        if "ich heisse " in msg_lower or "ich hei" in msg_lower:
            words = msg.split()
            # Very basic extraction: grab the last word
            if len(words) > 2:
                name = words[-1].strip(".,!?")
                state["user_name"] = name

        if (
            self._is_plain_stress_rating(msg)
            and state.get("last_intent") == "mental_checkin_start"
        ):
            results = [("mental_stress_support", 0.99)]
        else:
            results = self._klassifizieren(msg)

        if (
            not results
            and state.get("sleep_hours") is not None
            and state.get("last_context") == "mental"
        ):
            results = [("mental_sleep_support", 0.92)]

        if self._should_clarify_intent(results, follow_up):
            clarify_text = (
                "Ich bin mir noch nicht ganz sicher, was du gerade brauchst. "
                "Waehle bitte kurz die passende Richtung:"
            )
            return (
                self._format_response(
                    clarify_text,
                    state,
                    "",
                    self._clarification_buttons(results),
                    intent="thema_auswahl",
                ),
                None,
            )

        if results:
            primary_intent = self._choose_primary_intent(results, state, follow_up, msg)
            if primary_intent:
                primary_score = next(
                    (score for intent, score in results if intent == primary_intent),
                    float(results[0][1]),
                )
                remaining_results = [
                    (intent, score)
                    for intent, score in results
                    if intent != primary_intent
                ]
                results = [(primary_intent, primary_score)] + remaining_results

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
                    return self._build_intent_response(
                        active_exercise, state, steps[step], empathy_prefix, buttons
                    )
                state["active_exercise"] = None
                state["exercise_step"] = 0
                return self._build_intent_response(
                    active_exercise,
                    state,
                    "Gut gemacht! Wie fuehlst du dich jetzt?",
                    "",
                    [],
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
                    return self._build_intent_response(
                        active_exercise, state, steps[step], empathy_prefix, buttons
                    )
                state["active_exercise"] = None
                state["exercise_step"] = 0
                return self._build_intent_response(
                    active_exercise,
                    state,
                    "Sehr gut. Hat dich das ein wenig geerdet?",
                    "",
                    [],
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
                    return self._build_intent_response(
                        active_exercise, state, steps[step], empathy_prefix, buttons
                    )
                state["active_exercise"] = None
                state["exercise_step"] = 0
                return self._build_intent_response(
                    active_exercise,
                    state,
                    "Stark gemacht. Wenn du willst, machen wir als naechstes einen kurzen Check-in.",
                    "",
                    [],
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
            rotate_for_progression = (
                generic_follow_up and int(state.get("generic_follow_up_count", 0)) >= 2
            )
            follow_up_intent = self._pick_follow_up_intent(
                state,
                preferred_context=preferred_context or state.get("last_context"),
                rotate_for_progression=rotate_for_progression,
            )
            if follow_up_intent and (
                not top_intent or self._is_meta_intent(top_intent) or generic_follow_up
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
                    return self._build_intent_response(
                        follow_up_intent, state, response, empathy_prefix, []
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
                    return self._build_intent_response(
                        chosen_intent, state, response, empathy_prefix, []
                    )

        if results:
            for intent, _score in results:
                responses = self.intent_to_responses.get(intent, [])
                if responses:
                    response = self._pick_response(
                        intent,
                        prior_response if intent == state.get("last_intent") else None,
                    )
                    if response:
                        return self._build_intent_response(
                            intent, state, response, empathy_prefix, []
                        )

        if follow_up:
            follow_up_intent = self._pick_follow_up_intent(
                state,
                preferred_context=preferred_context,
                rotate_for_progression=(
                    generic_follow_up
                    and int(state.get("generic_follow_up_count", 0)) >= 2
                ),
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
                    return self._build_intent_response(
                        follow_up_intent, state, response, empathy_prefix, []
                    )

        return self._fallback_response(state, preferred_context)

    def _apply_response_mode(self, text: str, mode: str) -> str:
        if mode == "short":
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            compact = sentences[0] if sentences else text
            if len(compact) > 160:
                compact = compact[:157].rsplit(" ", 1)[0] + "..."
            return compact

        if mode == "detailed":
            if len(text) < 220:
                return (
                    f"{text} Wenn du willst, gehen wir den naechsten Schritt direkt "
                    "gemeinsam durch."
                )
        return text

    def _personalize_with_slots(
        self, text: str, state: Dict[str, Any], intent: str
    ) -> str:
        notes = []

        stress_level = state.get("stress_level")
        if intent == "mental_stress_support" and isinstance(stress_level, int):
            if stress_level >= 8:
                notes.append(
                    f"Du hast {stress_level}/10 Stress genannt. Bitte geh jetzt Schritt fuer Schritt vor."
                )
            elif stress_level <= 3:
                notes.append(
                    f"Mit {stress_level}/10 ist es schon etwas stabiler. Halte die kleinen Routinen aufrecht."
                )

        sleep_hours = state.get("sleep_hours")
        if intent == "mental_sleep_support" and isinstance(sleep_hours, (int, float)):
            if sleep_hours < 6:
                notes.append(
                    f"Du hast etwa {sleep_hours}h Schlaf genannt. Eine ruhige Abendroutine ist heute besonders wichtig."
                )
            else:
                notes.append(
                    f"Mit rund {sleep_hours}h Schlaf lohnt sich heute vor allem ein konstanter Schlafrhythmus."
                )

        if not notes:
            return text
        return f"{text} {' '.join(notes)}"

    def _topic_transition_prefix(self, state: Dict[str, Any], intent: str) -> str:
        previous_intent = state.get("last_intent", "")
        if (
            not intent
            or not previous_intent
            or previous_intent == intent
            or not self._is_mental_intent(previous_intent)
            or not self._is_mental_intent(intent)
        ):
            return ""

        current_turn = int(state.get("turns", 0))
        last_transition_turn = int(state.get("last_transition_turn", -999))
        transition_key = f"{previous_intent}->{intent}"

        # Do not spam transition phrases every turn.
        if current_turn - last_transition_turn < 2:
            return ""

        # Avoid repeating the same announced switch in short distance.
        if (
            state.get("last_transition_key") == transition_key
            and current_turn - last_transition_turn < 8
        ):
            return ""

        templates = [
            "Wenn es fuer dich passt, gehen wir von {from_topic} zu {to_topic}. ",
            "Lass uns kurz von {from_topic} Richtung {to_topic} schauen. ",
            "Okay, wir richten den Blick von {from_topic} auf {to_topic}. ",
            "Wir nehmen jetzt den Schritt von {from_topic} zu {to_topic}. ",
        ]
        template = self._pick_non_repeating_option(
            templates,
            state,
            "recent_transition_templates",
            window=3,
        )
        if not template:
            template = templates[0]

        state["last_transition_turn"] = current_turn
        state["last_transition_key"] = transition_key
        state["last_transition_template"] = template

        return template.format(
            from_topic=self._topic_label(previous_intent),
            to_topic=self._topic_label(intent),
        )

    def _format_response(
        self,
        text: str,
        state: Dict[str, Any],
        empathy_prefix: str,
        buttons: List[Dict[str, str]],
        intent: str = "",
    ) -> Dict[str, Any]:
        """Helper to inject placeholders, prepend empathy, and pack into UI dict."""
        uname = state.get("user_name", "")
        # Remove empty bracket placeholders if we don't know the name yet.
        if uname:
            text = text.replace("[Name]", uname)
        else:
            text = text.replace(" [Name]", "").replace("[Name]", "")

        transition_prefix = self._topic_transition_prefix(state, intent)

        final_text = (transition_prefix + empathy_prefix + text).strip()
        final_text = self._personalize_with_slots(final_text, state, intent)
        final_text = self._apply_response_mode(
            final_text, state.get("response_mode", DEFAULT_RESPONSE_MODE)
        )

        buttons = list(buttons or [])
        if any(button.get("value") == "weiter" for button in buttons):
            final_buttons = self._dedupe_buttons(buttons)
        elif state.get("active_exercise"):
            final_buttons = self._dedupe_buttons(
                buttons + [{"label": "Weiter", "value": "weiter"}]
            )
        else:
            final_buttons = self._dedupe_buttons(
                buttons + self._default_quick_buttons()
            )

        return {"text": final_text, "buttons": final_buttons}
