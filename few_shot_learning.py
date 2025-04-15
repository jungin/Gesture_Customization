import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("full_tcn_model.keras")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)