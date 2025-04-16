import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from variables import preprocessed_test_dir

model = load_model("full_tcn_model.keras")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

file = os.path.join(preprocessed_test_dir, '17.npy')
X_custom = np.load(file).reshape(1, 37, 63)
feature_vector = feature_extractor.predict(X_custom)  # shape: (1, 128)

def compute_prototypes(X_feat, y_labels):
    """
    X_feat: shape (K, D)
    y_labels: shape (K,)
    return: dict {label: prototype vector}
    """
    prototypes = {}
    for label in np.unique(y_labels):
        prototypes[label] = X_feat[y_labels == label].mean(axis=0)
    return prototypes
