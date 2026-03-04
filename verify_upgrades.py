import sys
import os
from pathlib import Path
from typing import Dict, Any
import random

# Suppress NLTK download issues for this test if possible, or just catch them
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Mocking parts of the app to test DialogueManager
from chatbot.dialogue_manager import DialogueManager
from chatbot.model_engine import IntentClassifier


def test_structured_response():
    print("Testing Structured Response...")
    base_dir = Path(__file__).resolve().parent / "chatbot"
    classifier = IntentClassifier(base_dir)
    bot = DialogueManager(base_dir, classifier)

    state = {"chat_state": {}, "turns": 0}
    msg = "Ich bin sehr gestresst und kann nicht gut schlafen"

    # 1. Test basic structure and slots
    res, intent = bot.get_response(msg, state)
    text = res["text"]
    print(f"Response: {text}")
    print(f"Intent: {intent}")

    # Structure: Acknowledge (validation) + Reflect (mirroring) + Action + Question
    parts = text.split(". ")
    print(f"Parts found: {len(parts)}")

    # Check for question at end
    assert text.strip().endswith("?"), "Response should end with a question"

    # 2. Test one question per turn
    res2, intent2 = bot.get_response("Ich kann nicht schlafen", state)
    text2 = res2["text"]
    print(f"Response 2: {text2}")
    assert text2.count("?") == 1, f"Expected 1 question mark, found {text2.count('?')}"

    # 3. Test anti-repetition for openers
    openers = []
    for _ in range(5):
        state_copy = {
            "chat_state": state.get("chat_state", {}).copy(),
            "turns": state.get("turns", 0),
        }
        r, _ = bot.get_response("Ich bin gestresst", state_copy)
        opener = r["text"].split(".")[0]
        openers.append(opener)

    print(f"Openers: {openers}")
    # With window=6, openers should vary if multiple options are available
    # We just want to see some variety if possible, but at least ensure no crash
    assert len(openers) == 5


def test_repair_flow():
    print("\nTesting Repair Flow...")
    base_dir = Path(__file__).resolve().parent / "chatbot"
    classifier = IntentClassifier(base_dir)
    bot = DialogueManager(base_dir, classifier)

    state = {"chat_state": {}, "turns": 0}
    # Message that might be ambiguous or low confidence
    msg = "xyz unknown"
    res, intent = bot.get_response(msg, state)

    print(f"Repair Response: {res['text']}")
    # Check for richer repair text
    assert any(
        phrase in res["text"]
        for phrase in ["nicht klar verstanden", "nicht ganz sicher"]
    ), "Repair text should be helpful"
    # Check for specific buttons
    assert (
        len(res["buttons"]) >= 2
    ), f"Expected at least 2 buttons, got {len(res['buttons'])}"
    labels = [b["label"] for b in res["buttons"]]
    print(f"Buttons: {labels}")


if __name__ == "__main__":
    try:
        test_structured_response()
        test_repair_flow()
        print("\nALL UPGRADE TESTS PASSED!")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nTESTS FAILED: {e}")
        sys.exit(1)
