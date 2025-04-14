import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utility import load_data

from variables import preprocessed_train_dir, preprocessed_val_dir,\
                      train_csv, val_csv, \
                      JESTER_CLASSES, \
                      preprocessed_train_subset_dir
                       

# path
# data_dir = preprocessed_train_dir                # .npy 파일들이 있는 경로
train_dir = preprocessed_train_subset_dir         # .npy 파일들이 있는 경로
val_dir = preprocessed_val_dir                    # .npy 파일들이 있는 경로
train_label_file = train_csv                         # 라벨 CSV 경로
val_label_file = val_csv                             # 라벨 CSV 경로


X_train, y_train = load_data(train_label_file, train_dir)
# X_val, y_val = load_data(val_label_file, val_dir)

import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(37, 63)),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(JESTER_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()



