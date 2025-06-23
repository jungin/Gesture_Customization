import os
from glob import glob
import tensorflow as tf
import numpy as np
from src.st_gcn import STGCN  # 위에서 만든 stgcn_tf.py
from src.variables import *  # JESTER_OUTPUT_DIR, etc.

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

meta = build_metadata_congd(CONGD_OUTPUT_DIR)
split_ratio = 0.9
split_idx = int(len(meta) * split_ratio)
train_meta = meta[:split_idx]  # 처음 90%를 train
val_meta   = meta[split_idx:]  # 나머지 10%를 validation
print(f"Train samples: {len(train_meta)}, Validation samples: {len(val_meta)}")

# 2) tf.data.Dataset 생성
def parse_fn(np_path, label):
    # np_path: bytes, so 먼저 문자열로 변환
    path = np_path.decode('utf-8')
    # (T, V, C) numpy 로드
    seq = np.load(path)  
    
     # 채널별로 평균0, 표준편차1
    mean = seq.mean(axis=(0,1), keepdims=True)
    std  = seq.std(axis=(0,1), keepdims=True)
    seq = (seq - mean) / (std + 1e-6)
 
   # (T, V, C) → (C, V, T)
    seq = np.transpose(seq, (2,1,0)).astype(np.float32)
    return seq, label

def tf_parse_fn(np_path, label):
    # tf.numpy_function을 사용해 numpy 로직 실행
    seq, lbl = tf.numpy_function(parse_fn, [np_path, label], [tf.float32, tf.int32])
    # shape 힌트 지정 (필수는 아니지만 좋음)
    seq.set_shape((3, 42, 37))   # C=3, V=42, T=37
    lbl.set_shape(())
    return seq, lbl

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

model = STGCN(in_channels=3, num_class=249, A=A, num_layers=6)  # ConGD

opt = tf.keras.optimizers.Adam(learning_rate=5e-7)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths))
train_ds = train_ds.map(tf_parse_fn, num_parallel_calls=AUTOTUNE)
# ← 여기서 .repeat() 추가
train_ds = train_ds.repeat()\
                   .batch(BATCH_SIZE)\
                   .prefetch(AUTOTUNE)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss   = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5)   if g is not None else None for g in grads]
    grads = [tf.clip_by_value(g, -0.1, 0.1) if g is not None else None for g in grads]
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

steps_per_epoch = len(train_paths) // BATCH_SIZE
# for epoch in range(EPOCHS):
#     for x_batch, y_batch in train_ds.take(steps_per_epoch):
#         l = train_step(x_batch, y_batch)
#     print(f"Epoch {epoch} | Loss: {l.numpy():.4f}")

# 1) Metric 객체 생성
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.SparseCategoricalAccuracy()

for epoch in range(EPOCHS):
    # --- Training ---
    train_acc_metric.reset_state()
    for x_batch, y_batch in train_ds.take(steps_per_epoch):
        # forward/backward & loss 계산
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss   = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # gradient clipping…
        opt.apply_gradients(zip(grads, model.trainable_variables))
        
        # Accuracy 업데이트
        train_acc_metric.update_state(y_batch, logits)
    
    train_acc = train_acc_metric.result().numpy()
    print(f"Epoch {epoch} — Loss: {loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # --- Validation ---
    val_acc_metric.reset_state()
    for x_batch, y_batch in val_ds:
        val_logits = model(x_batch, training=False)
        val_acc_metric.update_state(y_batch, val_logits)
    val_acc = val_acc_metric.result().numpy()
    print(f"           Val   Acc: {val_acc:.4f}")