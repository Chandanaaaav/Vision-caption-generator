import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Lazy-loading: model is loaded on first call, not at import time
_model = None
_tokenizer = None

max_length = 37

MODEL_PATH = os.path.join(os.path.dirname(__file__), "caption_model.keras")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")


def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
    return _tokenizer


def idx_to_word(integer):
    tokenizer = get_tokenizer()
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(feature):
    model = get_model()
    tokenizer = get_tokenizer()

    in_text = "startseq"

    while True:
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat)

        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    caption = in_text.replace("startseq", "").replace("endseq", "")

    return caption.strip()
