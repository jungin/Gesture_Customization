import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from variables import preprocessed_test_dir, train_csv

model = load_model("full_tcn_model.keras")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)


# 15: Swiping Down
# 9: Shaking Hand
# 12: Sliding Two Finger Right
new = [9, 12, 15]

# read train_csv
df = pd.read_csv(train_csv)

# extract three gestures
new_df = df[df['label_id'].isin(new)]

# pick five samples for each gesture
new_df = new_df.groupby('label_id').head(5)
files = new_df['video_id'].to_list()

# load the data
# X_test: shape (N, 37, 63) 
X_test = []
for file in files:
    file_path = os.path.join(preprocessed_test_dir, file + ".npy")
    X_test.append(np.load(file_path))
X_test = np.array(X_test)  # shape: (N, 37, 63)
X_test = X_test.reshape(-1, 37, 21*3)  # shape: (N, 37, 63)

# y_test: shape (N,)
y_test = []
for file in files:
    label_id = new_df[new_df['video_id'] == file]['label_id'].values[0]
    y_test.append(label_id)
y_test = np.array(y_test)  # shape: (N,)

feature_vector = feature_extractor.predict(X_test)  # shape: (1, 128)

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

# 프로토타입 계산
prototypes = compute_prototypes(feature_vector, y_test)

# 프로토타입 출력 (예: 각 클래스별 평균 임베딩 벡터)
for label, proto in prototypes.items():
    print("Label:", label, "Prototype shape:", proto.shape)

# 새로운 제스처에 대해 임베딩 추출 및 분류
new_gesture = np.load(os.path.join(preprocessed_test_dir, '505.npy')).reshape(1, 37, 63)  # 새 제스처 샘플
new_feature = feature_extractor.predict(new_gesture)  # (1, 128)

from scipy.spatial.distance import cdist
def predict_by_prototypes(query_feat, prototypes, metric='euclidean'):
    proto_matrix = np.stack(list(prototypes.values()))
    proto_labels = list(prototypes.keys())
    distances = cdist(query_feat, proto_matrix, metric=metric)
    pred_index = np.argmin(distances)
    return proto_labels[pred_index]

predicted_label = predict_by_prototypes(new_feature, prototypes)
print("Predicted custom gesture label:", predicted_label)