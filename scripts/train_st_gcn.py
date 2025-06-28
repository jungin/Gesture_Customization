import os
from glob import glob
import tensorflow as tf
import numpy as np
from src.st_gcn import STGCN  # 위에서 만든 stgcn_tf.py
from src.variables import *  # JESTER_OUTPUT_DIR, etc.
from src.utility import augment_features  # augment_features 함수가 있는 utility.py

inv = {v: k for k, v in JESTER_LABELS.items()}

# 1) 메타데이터 준비
def build_metadata_congd(root_dir, pattern="*.npy"):
    """
    root_dir 아래의 모든 .npy 파일을 찾고,
    파일명이나 디렉토리 구조에서 라벨을 추출하여
    [(path1, label1), (path2, label2), ...] 형태의 리스트를 반환.
    """
    meta = []
    for filepath in glob(os.path.join(root_dir, "**", pattern), recursive=True):
        fname = os.path.basename(filepath)
        name, _ = os.path.splitext(fname)           # "train(test)_folername_filename_framestart_end_lblXXX"
        label = name.split('_', 5)[-1][3:]          # "lblXXX" 
        meta.append((filepath, int(label) - 1))
    return meta

meta = build_metadata_congd(CONGD_OUTPUT_DIR + '.old')
split_ratio = 0.9
split_idx = int(len(meta) * split_ratio)
train_meta = meta[:split_idx]  # 처음 90%를 train
val_meta   = meta[split_idx:]  # 나머지 10%를 validation
print(f"Train samples: {len(train_meta)}, Validation samples: {len(val_meta)}")

# 2) tf.data.Dataset 생성
def parse_fn(np_path, label):
    path = np_path.decode('utf-8')
    seq = np.load(path)                 # (T, V, 3)

    # 1) 표준화
    mean = seq.mean(axis=(0,1), keepdims=True)
    std  = seq.std(axis=(0,1), keepdims=True)
    seq = (seq - mean) / (std + 1e-6)   # (T, V, 3)

    # 2) 롤·턴 방향성 피처 추가 (총 7채널)
    seq = augment_features(seq)         # (T, V, 7)

    # 3) (T, V, C) → (C, V, T)
    seq = np.transpose(seq, (2,1,0)).astype(np.float32)  # (7, V, T)
    return seq, label

def tf_parse_fn(np_path, label):
    seq, lbl = tf.numpy_function(parse_fn, [np_path, label],
                                 [tf.float32, tf.int32])
    # 채널 수를 7로 변경
    seq.set_shape((7, 42, 37))
    lbl.set_shape(())
    return seq, lbl


# # 2) tf.data.Dataset 생성
# def parse_fn(np_path, label):
#     # np_path: bytes, so 먼저 문자열로 변환
#     path = np_path.decode('utf-8')
#     # (T, V, C) numpy 로드
#     seq = np.load(path)  
    
#      # 채널별로 평균0, 표준편차1
#     mean = seq.mean(axis=(0,1), keepdims=True)
#     std  = seq.std(axis=(0,1), keepdims=True)
#     seq = (seq - mean) / (std + 1e-6)
 
#    # (T, V, C) → (C, V, T)
#     seq = np.transpose(seq, (2,1,0)).astype(np.float32)
#     return seq, label

# def tf_parse_fn(np_path, label):
#     # tf.numpy_function을 사용해 numpy 로직 실행
#     seq, lbl = tf.numpy_function(parse_fn, [np_path, label], [tf.float32, tf.int32])
#     # shape 힌트 지정 (필수는 아니지만 좋음)
#     seq.set_shape((3, 42, 37))   # C=3, V=42, T=37
#     lbl.set_shape(())
#     return seq, lbl

AUTOTUNE = tf.data.AUTOTUNE

# 문자열 리스트와 레이블 리스트로 변환
train_paths = [p for p, l in train_meta]
train_labels = [l for p, l in train_meta]
val_paths   = [p for p, l in val_meta]
val_labels  = [l for p, l in val_meta]

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths))
train_ds = train_ds.map(tf_parse_fn, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(tf_parse_fn, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 3) 모델 생성 및 컴파일
# 3.1) MediaPipe Hands 21개 랜드마크 인덱스에 따른 간선(edge) 목록
edges = [
    (0,1),(1,2),(2,3),(3,4),         # Thumb
    (0,5),(5,6),(6,7),(7,8),         # Index
    (0,9),(9,10),(10,11),(11,12),    # Middle
    (0,13),(13,14),(14,15),(15,16),  # Ring
    (0,17),(17,18),(18,19),(19,20)   # Pinky
]

V_hand = 21

# 3.2) 한 손 adjacency 초기화
A_hand = np.zeros((V_hand, V_hand), dtype=np.float32)
for i, j in edges:
    A_hand[i, j] = 1
    A_hand[j, i] = 1
np.fill_diagonal(A_hand, 1)  # self-loop

# 3.3) 정규화
D_hand = np.diag(1.0 / np.sqrt(A_hand.sum(axis=1)))
A_hand = D_hand @ A_hand @ D_hand

# 3.4) 두 손 합치기 (42×42 block-diagonal)
V = V_hand * 2
A = np.zeros((V, V), dtype=np.float32)
A[:V_hand, :V_hand] = A_hand            # 왼손 블록
A[V_hand:, V_hand:] = A_hand            # 오른손 블록

# 3.5) 양손 간 edge 추가 (ex: 손목과 손끝 연결)
# 손목 (0 ↔ 21)
A[0, 21] = A[21, 0] = 1

# 양손 손가락 끝끼리 연결 (Thumb~Pinky)
left_tips = [4, 8, 12, 16, 20]
right_tips = [25, 29, 33, 37, 41]
for l, r in zip(left_tips, right_tips):
    A[l, r] = A[r, l] = 1

# 3.6) 전체 정규화
D = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
A = D @ A @ D

model = STGCN(in_channels=7, num_class=249, A=A, num_layers=6)  # ConGD

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,  # 8 epoch 이상 성능 개선 없으면 중단
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'models/st_gcn_congd_best_model_add_channel.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 4) 학습
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop, checkpoint, lr_sched],
)

# 5) Feature Extractor 생성
# 5.1) blocks + global_pool 을 순차적으로 묶어 Feature Extractor 생성
feature_extractor = tf.keras.Sequential(
    model.blocks + [model.global_pool],
    name="stgcn_feature_extractor"
)

# 빈 리스트 준비
x_list, y_list = [], []

# val_ds 순회하며 배치별로 수집
for x_batch, y_batch in val_ds:
    x_list.append(x_batch.numpy())     # (batch_size, C, V, T)
    y_list.append(y_batch.numpy())     # (batch_size,)

# 리스트를 하나의 배열로 합치기
x_val = np.concatenate(x_list, axis=0)  # (N_val, C, V, T)
y_val = np.concatenate(y_list, axis=0)  # (N_val,)

# embedding 추출 
embeddings = feature_extractor(x_val, training=False)  # shape: (N_val, feature_dim)

import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix,
    classification_report
)

# 1) 정량적 클러스터링 지표 계산
sil_score = silhouette_score(embeddings, y_val)
ch_score  = calinski_harabasz_score(embeddings, y_val)
db_score  = davies_bouldin_score(embeddings, y_val)
print(f"Silhouette Score:        {sil_score:.4f}")
print(f"Calinski–Harabasz Score: {ch_score:.1f}")
print(f"Davies–Bouldin Score:    {db_score:.4f}\n")

# 2) UMAP 2D 시각화
reducer = umap.UMAP(n_components=2, random_state=42)
emb2d   = reducer.fit_transform(embeddings)  # (N_val, 2)

plt.figure(figsize=(8,8))
scatter = plt.scatter(
    emb2d[:,0], emb2d[:,1],
    c=y_val, s=5, cmap='tab20', alpha=0.8
)
plt.legend(*scatter.legend_elements(), title="Classes",
           bbox_to_anchor=(1.05,1))
plt.title("UMAP Projection of ST-GCN Embeddings")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()


# 1) Validation 데이터셋에서 y_true, y_pred 수집
y_true = []
y_pred = []

for x_batch, y_batch in val_ds:
    # 예측 (logits 또는 확률)
    logits = model.predict(x_batch, verbose=0)
    preds = np.argmax(logits, axis=1)
    
    y_pred.extend(preds.tolist())
    y_true.extend(y_batch.numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
num_classes = np.max(y_true) + 1


# 2) Confusion Matrix 계산 및 정규화 (행 기준)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# 3) Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, aspect='auto')   # 기본 colormap 사용
plt.colorbar()
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(num_classes), np.arange(num_classes), rotation=90)
plt.yticks(np.arange(num_classes), np.arange(num_classes))
plt.tight_layout()
plt.show()

# 4) Classification Report 출력
print("Classification Report:\n")
print(classification_report(y_true, y_pred, digits=4))



# opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
# train_ds = train_ds.shuffle(len(train_paths))
# train_ds = train_ds.map(tf_parse_fn, num_parallel_calls=AUTOTUNE)
# # ← 여기서 .repeat() 추가
# train_ds = train_ds.repeat()\
#                    .batch(BATCH_SIZE)\
#                    .prefetch(AUTOTUNE)

# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss   = loss_fn(y, logits)
#     grads = tape.gradient(loss, model.trainable_variables)
#     grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
#     opt.apply_gradients(zip(grads, model.trainable_variables))
#     train_acc_metric.update_state(y, logits)
#     return loss

# steps_per_epoch = len(train_paths) // BATCH_SIZE

# for lr in [1e-4, 3e-4, 1e-3]:
#     opt = tf.keras.optimizers.Adam(learning_rate=lr)
#     model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
#     print("Testing LR =", lr)
#     history = model.fit(train_ds.take(steps_per_epoch),
#                         validation_data=val_ds,
#                         epochs=3)
#     print()

# # 1) Metric 객체 생성
# train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
# val_acc_metric   = tf.keras.metrics.SparseCategoricalAccuracy()

# for epoch in range(EPOCHS):
#     # Training
#     train_acc_metric.reset_state()
#     for step, (x_batch, y_batch) in enumerate(train_ds.take(steps_per_epoch)):
#         loss = train_step(x_batch, y_batch)
#         if step % 100 == 0:
#             tf.print(f"epoch {epoch} step {step} loss {loss:.4f}")
#     train_acc = train_acc_metric.result()
#     tf.print(f"Epoch {epoch} Train Acc: {train_acc:.4f}")
    
#     # --- Validation ---
#     val_acc_metric.reset_state()
#     for x_batch, y_batch in val_ds:
#         val_logits = model(x_batch, training=False)
#         val_acc_metric.update_state(y_batch, val_logits)
#     val_acc = val_acc_metric.result().numpy()
#     print(f"           Val   Acc: {val_acc:.4f}")