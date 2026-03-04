import json
import unittest
from unittest.mock import patch

from chatbot.app import DEFAULT_RESPONSE, app, bot


class ChatbotGetEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _answers_for_intent(self, intent):
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == intent:
                return item["antwort"]
        return []

    def _response_contains_intent_answer(self, response_text, intent):
        answers = self._answers_for_intent(intent)
        for answer in answers:
            clean_answer = answer.replace(" [Name]", "").replace("[Name]", "")
            if clean_answer in response_text:
                return True
        return False

    def _response_contains_any_intent_answer(self, response_text, intents):
        return any(
            self._response_contains_intent_answer(response_text, intent)
            for intent in intents
        )

    def test_get_returns_response(self):
        response = self.client.post("/get", data={"msg": "Hallo"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("text", data)
        self.assertTrue(data["text"])

    def test_get_empty_message(self):
        response = self.client.post("/get", data={"msg": ""})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["text"], "")

    def test_get_fallback_response_on_internal_error(self):
        with patch.object(bot, "get_response", side_effect=Exception("boom")):
            response = self.client.post("/get", data={"msg": "Hallo"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["text"], DEFAULT_RESPONSE)

    def test_get_unknown_message_returns_default(self):
        response = self.client.post("/get", data={"msg": "Banane Fahrrad Wolke"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["text"], DEFAULT_RESPONSE)
        self.assertIn("buttons", data)
        self.assertGreaterEqual(len(data["buttons"]), 1)

    def test_get_stimmung_question_returns_stimmung_answer(self):
        response = self.client.post("/get", data={"msg": "Wie geht es dir?"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "smalltalk_stimmung")
        )

    def test_help_command_returns_thema_auswahl(self):
        response = self.client.post("/get", data={"msg": "/hilfe"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "thema_auswahl")
        )

    def test_follow_up_detail_request_after_mental_prompt(self):
        first = self.client.post("/get", data={"msg": "Ich bin gestresst"})
        self.assertEqual(first.status_code, 200)
        second = self.client.post("/get", data={"msg": "erklaer genauer"})
        self.assertEqual(second.status_code, 200)
        data = json.loads(second.data)
        self.assertNotEqual(data["text"], DEFAULT_RESPONSE)

    def test_flexible_focus_intent_detection(self):
        response = self.client.post(
            "/get", data={"msg": "Ich bin gerade total abgelenkt und komme nicht ins Tun"}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_focus_support")
        )

    def test_keyword_energy_intent_detection(self):
        response = self.client.post(
            "/get", data={"msg": "Heute fuehle ich mich komplett antriebslos und energielos"}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_energy_support")
        )

    def test_overthinking_intent_detection(self):
        response = self.client.post(
            "/get", data={"msg": "Ich habe gerade ein Gedankenkarussell"}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(
                data["text"], "mental_overthinking_support"
            )
        )

    def test_crisis_keyword_routes_to_crisis_intent(self):
        response = self.client.post("/get", data={"msg": "Ich will mir was antun"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_crisis_support")
        )

    def test_body_scan_follow_up_steps(self):
        first = self.client.post("/get", data={"msg": "Body Scan bitte"})
        self.assertEqual(first.status_code, 200)
        first_data = json.loads(first.data)
        self.assertTrue(
            self._response_contains_intent_answer(first_data["text"], "mental_body_scan")
        )

        second = self.client.post("/get", data={"msg": "weiter"})
        self.assertEqual(second.status_code, 200)
        second_data = json.loads(second.data)
        self.assertIn("Schritt 1", second_data["text"])
        self.assertIn("buttons", second_data)
        self.assertTrue(any(btn.get("value") == "weiter" for btn in second_data["buttons"]))

    def test_sleep_slot_personalization(self):
        response = self.client.post(
            "/get", data={"msg": "Ich kann nicht schlafen, nur 4 Stunden"}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_sleep_support")
        )
        self.assertIn("4.0h", data["text"])

    def test_explicit_breathing_prompt_not_overridden_by_follow_up_marker(self):
        response = self.client.post("/get", data={"msg": "Noch eine Atemuebung"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(
                data["text"], "mental_breathing_exercise"
            )
        )

    def test_plain_check_in_maps_to_mental_checkin(self):
        response = self.client.post("/get", data={"msg": "check in"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_checkin_start")
        )

    def test_low_confidence_triggers_clarification(self):
        mocked_results = [
            ("mental_stress_support", 0.41),
            ("mental_sleep_support", 0.37),
        ]
        with patch.object(bot, "_klassifizieren", return_value=mocked_results):
            response = self.client.post("/get", data={"msg": "Unklar"})

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("nicht ganz sicher", data["text"])
        self.assertTrue(any(btn.get("value") == "Ich bin gestresst" for btn in data["buttons"]))

    def test_response_mode_short(self):
        mode_response = self.client.post("/get", data={"msg": "/modus kurz"})
        self.assertEqual(mode_response.status_code, 200)
        mode_data = json.loads(mode_response.data)
        self.assertIn("kuerzer", mode_data["text"])

        response = self.client.post("/get", data={"msg": "Ich bin gestresst"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertLess(len(data["text"]), 170)

    def test_generic_follow_up_keeps_topic_on_first_turn(self):
        first = self.client.post("/get", data={"msg": "Ich bin gestresst"})
        self.assertEqual(first.status_code, 200)

        second = self.client.post("/get", data={"msg": "weiter bitte"})
        self.assertEqual(second.status_code, 200)
        second_data = json.loads(second.data)
        self.assertTrue(
            self._response_contains_intent_answer(
                second_data["text"], "mental_stress_support"
            )
        )

    def test_repeated_generic_follow_up_progresses_to_next_support_step(self):
        first = self.client.post("/get", data={"msg": "Ich bin gestresst"})
        self.assertEqual(first.status_code, 200)

        second = self.client.post("/get", data={"msg": "weiter"})
        self.assertEqual(second.status_code, 200)
        second_data = json.loads(second.data)
        self.assertTrue(
            self._response_contains_intent_answer(
                second_data["text"], "mental_stress_support"
            )
        )

        third = self.client.post("/get", data={"msg": "weiter"})
        self.assertEqual(third.status_code, 200)
        third_data = json.loads(third.data)
        self.assertTrue(
            self._response_contains_any_intent_answer(
                third_data["text"],
                [
                    "mental_breathing_exercise",
                    "mental_grounding",
                    "mental_focus_support",
                ],
            )
        )

    def test_repeated_unknown_messages_trigger_repair_prompt(self):
        first = self.client.post("/get", data={"msg": "xyz qwertz asdf"})
        self.assertEqual(first.status_code, 200)
        first_data = json.loads(first.data)
        self.assertEqual(first_data["text"], DEFAULT_RESPONSE)

        second = self.client.post("/get", data={"msg": "blub blub blub"})
        self.assertEqual(second.status_code, 200)
        second_data = json.loads(second.data)
        self.assertIn("noch nicht klar verstanden", second_data["text"])
        self.assertIn("buttons", second_data)
        self.assertGreaterEqual(len(second_data["buttons"]), 1)

    def test_topic_switch_uses_transition_phrase(self):
        first = self.client.post("/get", data={"msg": "Ich bin gestresst"})
        self.assertEqual(first.status_code, 200)

        second = self.client.post("/get", data={"msg": "Ich habe Angst"})
        self.assertEqual(second.status_code, 200)
        second_data = json.loads(second.data)
        self.assertIn("Stress", second_data["text"])
        self.assertIn("Angst", second_data["text"])
        self.assertTrue(
            self._response_contains_intent_answer(
                second_data["text"], "mental_anxiety_support"
            )
        )

    def test_response_still_works_when_textblobde_is_unavailable(self):
        with patch("chatbot.dialogue_manager.TextBlobDE", None):
            response = self.client.post(
                "/get", data={"msg": "Das ist heute wirklich schwierig fuer mich"}
            )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("text", data)
        self.assertTrue(data["text"])


if __name__ == "__main__":
    unittest.main()
