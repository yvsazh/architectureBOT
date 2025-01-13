import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime
import pathlib

img_height = 299
img_width = 299
num_classes = 25  # Залежить від вашого датасету

class_names = ['Achaemenid architecture', 'American Foursquare architecture', 'American craftsman style', 'Ancient Egyptian architecture', 'Art Deco architecture', 'Art Nouveau architecture', 'Baroque architecture', 'Bauhaus architecture', 'Beaux-Arts architecture', 'Byzantine architecture', 'Chicago school architecture', 'Colonial architecture', 'Deconstructivism', 'Edwardian architecture', 'Georgian architecture', 'Gothic architecture', 'Greek Revival architecture', 'International style', 'Novelty architecture', 'Palladian architecture', 'Postmodern architecture', 'Queen Anne architecture', 'Romanesque architecture', 'Russian Revival architecture', 'Tudor Revival architecture']

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Заморозити базову модель

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # Останній шар
])

model.load_weights("./savedModels/architecture_style_classifier_weights.h5")

import os
import requests
from io import BytesIO

def predictOnImage(image_url):
    try:
        # Завантаження зображення через посилання
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            image_data = BytesIO(response.content)

            # Відкриваємо зображення через PIL
            img = PIL.Image.open(image_data)
            img = img.convert("RGB")  # Конвертуємо до RGB, якщо необхідно
            img = img.resize((img_width, img_height))

            # Перетворення зображення в формат, що підходить для моделі
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Додаємо batch dimension

            # Робимо передбачення
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            # Виводимо результат
            print("Будівля побудована у стилі {} ({:.2f}% вероятность)".format(
                class_names[np.argmax(score)],
                100 * np.max(score)))

            # Показуємо зображення
            img.show()
        else:
            print(f"Не вдалося завантажити зображення. Код відповіді: {response.status_code}")

    except Exception as e:
        print(f"Виникла помилка при обробці зображення: {e}")

predictOnImage(input("please: "))