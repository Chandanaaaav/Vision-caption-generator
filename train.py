import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

# -----------------------------
# Load files
# -----------------------------

df = pd.read_csv("cleaned_captions.xls")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("image_features.pkl", "rb") as f:
    features = pickle.load(f)

print("Total Features:", len(features))

# -----------------------------
# Vocabulary
# -----------------------------

vocab_size = len(tokenizer.word_index) + 1

captions = df["caption"].tolist()

max_length = max(len(caption.split()) for caption in captions)

print("Vocabulary Size:", vocab_size)
print("Maximum Caption Length:", max_length)

# -----------------------------
# Mapping
# -----------------------------

mapping = {}

for _, row in df.iterrows():

    image = row["image"]
    caption = row["caption"]

    if image not in mapping:
        mapping[image] = []

    mapping[image].append(caption)

print("Total Images:", len(mapping))
class DataGenerator(Sequence):

    def __init__(self, mapping, features, tokenizer, max_length, vocab_size, batch_size=32):

        self.mapping = mapping
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.image_ids = list(mapping.keys())

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, index):

        batch_ids = self.image_ids[index*self.batch_size:(index+1)*self.batch_size]

        X1, X2, y = [], [], []

        for img_id in batch_ids:

            feature = self.features[img_id]

            captions = self.mapping[img_id]

            for caption in captions:

                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):

                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]

                    out_seq = to_categorical(out_seq, num_classes=self.vocab_size)

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        return (np.array(X1), np.array(X2)), np.array(y)

train_generator = DataGenerator(
    mapping,
    features,
    tokenizer,
    max_length,
    vocab_size,
    batch_size=32
)

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = Add()([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)

outputs = Dense(vocab_size, activation="softmax")(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)
history = model.fit(
    train_generator,
    epochs=20
)

model.save("caption_model.keras")
