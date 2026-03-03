import unittest
from unittest.mock import patch

from chatbot.app import DEFAULT_RESPONSE, app, dialogflow


class ChatbotGetEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_get_returns_response(self):
        response = self.client.post("/get", data={"msg": "Hallo"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.decode("utf-8").strip())

    def test_get_empty_message(self):
        response = self.client.post("/get", data={"msg": ""})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode("utf-8"), "")

    def test_get_fallback_response_on_internal_error(self):
        with patch("chatbot.app.antwort", side_effect=RuntimeError("boom")):
            response = self.client.post("/get", data={"msg": "Hallo"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode("utf-8"), DEFAULT_RESPONSE)

    def test_get_unknown_message_returns_default(self):
        response = self.client.post("/get", data={"msg": "Banane Fahrrad Wolke"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode("utf-8"), DEFAULT_RESPONSE)

    def test_get_stimmung_question_returns_stimmung_answer(self):
        response = self.client.post("/get", data={"msg": "Wie geht es dir?"})
        self.assertEqual(response.status_code, 200)
        body = response.data.decode("utf-8")
        stimmung_answers = []
        for item in dialogflow["dialogflow"]:
            if item["intent"] == "smalltalk_stimmung":
                stimmung_answers = item["antwort"]
                break
        self.assertIn(body, stimmung_answers)

    def test_follow_up_reuses_context(self):
        first = self.client.post("/get", data={"msg": "Erzaehl einen Witz"})
        self.assertEqual(first.status_code, 200)
        second = self.client.post("/get", data={"msg": "noch einen"})
        self.assertEqual(second.status_code, 200)
        self.assertNotEqual(second.data.decode("utf-8"), DEFAULT_RESPONSE)

    def test_follow_up_detail_request_after_philosophy_prompt(self):
        first = self.client.post("/get", data={"msg": "Stoiker"})
        self.assertEqual(first.status_code, 200)
        second = self.client.post("/get", data={"msg": "erklaer genauer"})
        self.assertEqual(second.status_code, 200)
        body = second.data.decode("utf-8")
        self.assertNotEqual(body, DEFAULT_RESPONSE)
        feedback_positive_answers = []
        for item in dialogflow["dialogflow"]:
            if item["intent"] == "feedback_positiv":
                feedback_positive_answers = item["antwort"]
                break
        self.assertNotIn(body, feedback_positive_answers)

    def test_explicit_schopenhauer_prompt_not_overridden_by_follow_up_marker(self):
        response = self.client.post("/get", data={"msg": "Erklaer Schopenhauer"})
        self.assertEqual(response.status_code, 200)
        body = response.data.decode("utf-8")
        schopenhauer_answers = []
        for item in dialogflow["dialogflow"]:
            if item["intent"] == "philosophie_schopenhauer":
                schopenhauer_answers = item["antwort"]
                break
        self.assertIn(body, schopenhauer_answers)

    def test_explicit_wortspiel_prompt_not_overridden_by_follow_up_marker(self):
        response = self.client.post("/get", data={"msg": "Noch ein Wortspiel"})
        self.assertEqual(response.status_code, 200)
        body = response.data.decode("utf-8")
        wortspiel_answers = []
        for item in dialogflow["dialogflow"]:
            if item["intent"] == "fun_wortspiel":
                wortspiel_answers = item["antwort"]
                break
        self.assertIn(body, wortspiel_answers)


if __name__ == "__main__":
    unittest.main()
