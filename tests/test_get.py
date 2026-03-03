import unittest

from chatbot.app import app


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


if __name__ == "__main__":
    unittest.main()
