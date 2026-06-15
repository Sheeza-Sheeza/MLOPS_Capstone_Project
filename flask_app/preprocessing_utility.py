import re
import string

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

_NLTK_READY = False
_LEMMATIZER = None
_STOP_WORDS = None


def ensure_nltk_resources() -> None:
    """Download and eagerly load NLTK corpora used for inference."""
    global _NLTK_READY, _LEMMATIZER, _STOP_WORDS

    if _NLTK_READY:
        return

    for package in ("wordnet", "omw-1.4", "stopwords"):
        nltk.download(package, quiet=True)

    # Force-load corpora before Flask worker threads use them.
    stopwords.words("english")
    wordnet.synsets("test")

    _LEMMATIZER = WordNetLemmatizer()
    _STOP_WORDS = set(stopwords.words("english"))
    _NLTK_READY = True


def preprocess_text(text):
    """Match the training pipeline in src/data/data_preprocessing.py."""
    ensure_nltk_resources()

    if not isinstance(text, str):
        return ""

    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = "".join(char for char in text if not char.isdigit())
    text = text.lower()
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(word for word in text.split() if word not in _STOP_WORDS)
    text = " ".join(_LEMMATIZER.lemmatize(word) for word in text.split())
    return text


ensure_nltk_resources()
