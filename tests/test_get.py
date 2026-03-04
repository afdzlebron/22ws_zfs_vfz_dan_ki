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

    def test_explicit_breathing_prompt_not_overridden_by_follow_up_marker(self):
        response = self.client.post("/get", data={"msg": "Noch eine Atemuebung"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_breathing_exercise")
        )

    def test_plain_check_in_maps_to_mental_checkin(self):
        response = self.client.post("/get", data={"msg": "check in"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(
            self._response_contains_intent_answer(data["text"], "mental_checkin_start")
        )


if __name__ == "__main__":
    unittest.main()
