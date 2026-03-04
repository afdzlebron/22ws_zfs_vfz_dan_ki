import unittest
from unittest.mock import patch
import json

from chatbot.app import DEFAULT_RESPONSE, app, bot


class ChatbotGetEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

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
        # Patching bot.get_response since app.py calls it
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

    def test_get_stimmung_question_returns_stimmung_answer(self):
        response = self.client.post("/get", data={"msg": "Wie geht es dir?"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        text = data["text"]

        stimmung_answers = []
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == "smalltalk_stimmung":
                stimmung_answers = item["antwort"]
                break

        # The bot might sanitize/format the answer (e.g. replacing [Name])
        # We check if the core part of the answer is present
        found = False
        for answer in stimmung_answers:
            clean_answer = answer.replace(" [Name]", "").replace("[Name]", "")
            if clean_answer in text:
                found = True
                break
        self.assertTrue(
            found, f"Response '{text}' not found in expected answers {stimmung_answers}"
        )

    def test_follow_up_reuses_context(self):
        first = self.client.post("/get", data={"msg": "Bewerte ein Produkt"})
        self.assertEqual(first.status_code, 200)
        second = self.client.post("/get", data={"msg": "noch genauer"})
        self.assertEqual(second.status_code, 200)
        data = json.loads(second.data)
        self.assertNotEqual(data["text"], DEFAULT_RESPONSE)

    def test_follow_up_detail_request_after_mental_prompt(self):
        first = self.client.post("/get", data={"msg": "Ich bin gestresst"})
        self.assertEqual(first.status_code, 200)
        second = self.client.post("/get", data={"msg": "erklaer genauer"})
        self.assertEqual(second.status_code, 200)
        data = json.loads(second.data)
        self.assertNotEqual(data["text"], DEFAULT_RESPONSE)

        feedback_positive_answers = []
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == "feedback_positiv":
                feedback_positive_answers = item["antwort"]
                break
        self.assertNotIn(data["text"], feedback_positive_answers)

    def test_explicit_ecoscore_prompt_not_overridden_by_follow_up_marker(self):
        response = self.client.post("/get", data={"msg": "Erklaer Eco-Score"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        eco_score_answers = []
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == "sustainable_eco_score_explain":
                eco_score_answers = item["antwort"]
                break
        self.assertIn(data["text"], eco_score_answers)

    def test_explicit_breathing_prompt_not_overridden_by_follow_up_marker(self):
        response = self.client.post("/get", data={"msg": "Noch eine Atemuebung"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        breathing_answers = []
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == "mental_breathing_exercise":
                breathing_answers = item["antwort"]
                break
        self.assertIn(data["text"], breathing_answers)

    def test_plain_check_in_maps_to_mental_checkin(self):
        response = self.client.post("/get", data={"msg": "check in"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        checkin_answers = []
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == "mental_checkin_start":
                checkin_answers = item["antwort"]
                break
        self.assertIn(data["text"], checkin_answers)

    def test_plain_materialvergleich_maps_to_materials_intent(self):
        response = self.client.post("/get", data={"msg": "Materialvergleich"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        material_answers = []
        for item in bot.dialogflow["dialogflow"]:
            if item["intent"] == "sustainable_materials":
                material_answers = item["antwort"]
                break
        self.assertIn(data["text"], material_answers)


if __name__ == "__main__":
    unittest.main()
