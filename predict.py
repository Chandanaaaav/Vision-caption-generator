import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("caption_model.keras")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

max_length = 37


def idx_to_word(integer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(feature):

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

    caption = in_text.replace("startseq","").replace("endseq","")

    return caption.strip()