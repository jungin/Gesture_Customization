import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from variables import preprocessed_test_dir, train_csv

def load_data(files, df):
    X, y = [], []
    for file in files:
        file_path = os.path.join(preprocessed_test_dir, str(file) + ".npy")
        X.append(np.load(file_path))
        
        label_id = df[df['video_id'] == file]['label_id'].values[0]
        y.append(label_id)

    X = np.array(X)  # shape: (N, 37, 63)
    X = X.reshape(-1, 37, 21*3)  # shape: (N, 37, 63)
    y = np.array(y)  # shape: (N,)

    return X, y

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

def predict_by_prototypes(query_feat, prototypes, metric='euclidean'):
    proto_matrix = np.stack(list(prototypes.values()))
    proto_labels = list(prototypes.keys())
    distances = cdist(query_feat, proto_matrix, metric=metric)
    pred_index = np.argmin(distances)
    return proto_labels[pred_index]


model = load_model("full_tcn_model.keras")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# 15: Swiping Down
# 9: Shaking Hand
# 12: Sliding Two Finger Right
new = [9, 12, 15]

# read train_csv
df = pd.read_csv(train_csv)

# extract three gestures
custom_df = df[df['label_id'].isin(new)]

# pick five samples for each gesture
custom_df = custom_df.groupby('label_id').head(5)
files = custom_df['video_id'].to_list()

X_custom, y_custom = load_data(files, custom_df)
feature_vector = feature_extractor.predict(X_custom)  # shape: (1, 128)

# 프로토타입 계산
prototypes = compute_prototypes(feature_vector, y_custom)

# 프로토타입 출력 (예: 각 클래스별 평균 임베딩 벡터)
for label, proto in prototypes.items():
    print("Label:", label, "Prototype shape:", proto.shape)

# extract three gestures
# pick five samples for each gesture
test_df = df[df['label_id'].isin(new)] \
           .groupby('label_id', group_keys=False) \
           .sample(n=100, random_state=42)
files = test_df['video_id'].to_list()

X_test, y_test = load_data(files, test_df)
y_pred = []
for i in range(len(X_test)):
    feat = feature_extractor.predict(X_test[i:i+1])
    pred = predict_by_prototypes(feat, prototypes)
    y_pred.append(pred)

# 실제 라벨과 예측 라벨
y_true = y_test  # y_test는 이미 정수형 라벨로 되어 있음

# Calculate evaluation metrics
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=[str(label) for label in np.unique(y_true)])

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)