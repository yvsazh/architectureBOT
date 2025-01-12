import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime

import pathlib

if tf.test.is_gpu_available():
	print("I am using GPU!!!")
else:
	print("I am NOT using GPU!!!")

dataset_dir = pathlib.Path("./architectural-styles-dataset/")
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

batch_size = 32
img_width = 299
img_height = 299

train_ds = tf.keras.utils.image_dataset_from_directory(
	dataset_dir,
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
	dataset_dir,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create model
num_classes = len(class_names)

base_model = tf.keras.applications.EfficientNetB7(
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

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])


model.summary()

epochs = 25

# checkpoint_dir = "./checkpoints"
# checkpoint_path = f"{checkpoint_dir}/model_epoch{{epoch:02d}}_val_loss{{val_loss:.2f}}"

# # Callbacks для збереження моделі
# checkpoint_callback = ModelCheckpoint(
#     filepath=checkpoint_path,      # Куди зберігати модель
#     monitor='val_loss',            # Моніторинг метрики (наприклад, валідаційної втрати)
#     save_best_only=True,           # Зберігати лише найкращу модель
#     save_weights_only=False,       # Зберігати всю модель (ваги + архітектуру)
#     mode='min',                    # Найкраще значення — мінімум
#     verbose=1                      # Вивід прогресу в консоль
# )

# Навчання
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback]
)

model.save_weights("./savedModels/architecture_style_classifier_weights_b7.h5")
