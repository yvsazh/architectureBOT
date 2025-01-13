import numpy as np
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
num_classes = 48  # Залежить від вашого датасету

class_names_en = [
    'Achaemenid architecture',
    'American Foursquare architecture',
    'American craftsman style',
    'Ancient Egyptian architecture',
    'Art Deco architecture',
    'Art Nouveau architecture',
    'Baroque architecture',
    'Bauhaus architecture',
    'Beaux-Arts architecture',
    'Byzantine architecture',
    'Chicago school architecture',
    'Colonial architecture',
    'Deconstructivism',
    'Edwardian architecture',
    'Georgian architecture',
    'Gothic architecture',
    'Greek Revival architecture',
    'International style',
    'Novelty architecture',
    'Palladian architecture',
    'Postmodern architecture',
    'Queen Anne architecture',
    'Romanesque architecture',
    'Russian Revival architecture',
    'Tudor Revival architecture'
]

class_names_ua = [
    'Ахеменідська архітектура',
    'Американська чотирикутна архітектура',
    'Американський ремісничий стиль',
    'Архітектура Стародавнього Єгипту',
    'Ар-деко',
    'Модерн',
    'Бароко',
    'Баухаус',
    'Боз-ар',
    'Візантійська архітектура',
    'Архітектура чиказької школи',
    'Колоніальна архітектура',
    'Деконструктивізм',
    'Едвардіанська архітектура',
    'Георгіанська архітектура',
    'Готика',
    'Грецьке відродження',
    'Інтернаціональний стиль',
    'Нестандартна архітектура',
    'Палладіанська архітектура',
    'Постмодерн',
    'Архітектура королеви Анни',
    'Романська архітектура',
    'Російське відродження',
    'Тюдорівське відродження'
]

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')  # Останній шар
])

model.load_weights("./architecture_style_classifier_weights_v2.h5")