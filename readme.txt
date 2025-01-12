tensorboard --logdir logs/fit

from tensorflow.keras.models import load_model
model = load_model("path_to_model_directory")