import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utility import load_data

import tensorflow as tf
from tensorflow.keras import layers, models

from variables import preprocessed_train_dir, preprocessed_val_dir,\
                      train_csv, val_csv, \
                      JESTER_CLASSES, \
                      preprocessed_train_subset_dir, preprocessed_val_subset_dir
                       

# path
train_dir = preprocessed_train_dir                # .npy 파일들이 있는 경로
val_dir = preprocessed_val_dir                    # .npy 파일들이 있는 경로
# train_dir = preprocessed_train_subset_dir         # .npy 파일들이 있는 경로
# val_dir = preprocessed_val_subset_dir                    # .npy 파일들이 있는 경로
train_label_file = train_csv                         # 라벨 CSV 경로
val_label_file = val_csv                             # 라벨 CSV 경로


X_train, y_train = load_data(train_label_file, train_dir)
X_val, y_val = load_data(val_label_file, val_dir)

def build_tcn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)  # (37, 63)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=1)(inputs)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=2)(x)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=4)(x)
    x = layers.GlobalAveragePooling1D()(x)     # ← 이게 우리가 뽑을 feature야!
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

X_train = X_train.reshape(-1, 37, 21*3)  # 즉, (samples, 37, 63)로 변경
X_val = X_val.reshape(-1, 37, 21*3)  # 즉, (samples, 37, 63)로 변경

model = build_tcn_model((37, 21*3), JESTER_CLASSES)
model.compile(optimizer='adamW', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))






