import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from typing import List, Set
import numpy as np

logger = logging.getLogger(__name__)

stemmer = GermanStemmer()


def ensure_nltk_data(nltk_data_dir: str) -> None:
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    import os

    os.makedirs(nltk_data_dir, exist_ok=True)
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
    }
    for resource, name in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(name, download_dir=nltk_data_dir)
            except Exception as exc:
                logger.warning("Failed to download NLTK resource %s: %s", name, exc)


def get_ignore_words() -> Set[str]:
    try:
        stop_words = set(stopwords.words("german"))
    except LookupError as exc:
        logger.warning("NLTK stopwords not available yet: %s", exc)
        stop_words = set()
    return {"?", ".", ","}.union(stop_words)


def normalize_phrase(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return " ".join(normalized.split())


def frage_bearbeitung(frage: str, ignore_words: Set[str]) -> List[str]:
    try:
        sentence_word = nltk.word_tokenize(frage, language="german")
    except LookupError as exc:
        logger.warning("NLTK punkt not available, falling back to split: %s", exc)
        sentence_word = frage.split()

    sentence_words = []
    for word in sentence_word:
        if word not in ignore_words or word in {"weiter", "andere", "nicht"}:
            sentence_words.append(word)

    return [stemmer.stem(word.lower()) for word in sentence_words]


def bow(frage: str, words: List[str], ignore_words: Set[str]) -> np.ndarray:
    sentence_words = frage_bearbeitung(frage, ignore_words)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)
