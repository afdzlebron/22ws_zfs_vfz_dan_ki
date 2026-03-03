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


if __name__ == "__main__":
    unittest.main()
