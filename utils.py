import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

base_model = InceptionV3(weights="imagenet")
model = Model(base_model.input, base_model.layers[-2].output)

def extract_feature(image_path):
    image = load_img(image_path, target_size=(299,299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)

    return feature