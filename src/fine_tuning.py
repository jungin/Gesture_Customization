import tensorflow as tf
import numpy as np
from src.variables import *
from src.utility import build_metadata_jester, tf_parse_fn
from src.st_gcn import STGCN  # 위에서 만든 stgcn_tf.py

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
# self-loop 추가
np.fill_diagonal(A_hand, 1)

# 3.3) 정규화
D = np.diag(1.0 / np.sqrt(A_hand.sum(axis=1)))
A_hand = D @ A_hand @ D

# 3.4) 두 손 합치기 (42×42 block-diagonal)
V = V_hand * 2
A = np.zeros((V, V), dtype=np.float32)
A[:V_hand, :V_hand] = A_hand      # 왼손 블록
A[V_hand:, V_hand:] = A_hand      # 오른손 블록

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


# 1) 모델 인스턴스 생성
model = STGCN(in_channels=7, num_class=27, A=A, num_layers=9)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 2) build() 혹은 dummy 호출로 내부 레이어들 생성
#    → 이 단계에서 모든 GraphConvolution, BatchNorm 등 레이어가 build 됩니다.
_ = model(tf.zeros((1, 7, 42, 37)), training=False)

# 3) weights 로드
model.load_weights('models/st_gcn_jester_best_model_add_channel.keras')

# 4) Feature Extractor 생성 & build
feature_extractor = tf.keras.Sequential(model.blocks + [model.global_pool])
_ = feature_extractor(tf.zeros((1, 7, 42, 37)), training=False)


AUTOTUNE = tf.data.AUTOTUNE
jester_root = JESTER_OUTPUT_DIR    # .npy들이 모여 있는 최상위 폴더
train_meta = build_metadata_jester(os.path.join(jester_root, "Train"))
val_meta   = build_metadata_jester(os.path.join(jester_root, "Validation"))
print(f"Train samples: {len(train_meta)}, Validation samples: {len(val_meta)}")


# 문자열 리스트와 레이블 리스트로 변환
val_paths   = [p for p, l in val_meta]
val_labels  = [l for p, l in val_meta]

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(tf_parse_fn, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# 빈 리스트 준비
x_list, y_list = [], []

# val_ds 순회하며 배치별로 수집
for x_batch, y_batch in val_ds:
    x_list.append(x_batch.numpy())     # (batch_size, C, V, T)
    y_list.append(y_batch.numpy())     # (batch_size,)

# 리스트를 하나의 배열로 합치기
x_val = np.concatenate(x_list, axis=0)  # (N_val, C, V, T)
y_val = np.concatenate(y_list, axis=0)  # (N_val,)

# 5) 이후 x_val 준비 → 임베딩 추출 → numpy 변환 → silhouette, UMAP...
emb_tensor  = feature_extractor(tf.constant(x_val), training=False)
embeddings  = emb_tensor.numpy()