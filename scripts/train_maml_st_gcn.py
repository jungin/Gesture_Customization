import os
import random
import numpy as np
import tensorflow as tf
from glob import glob
from src.st_gcn import STGCN
from src.data_loader import GestureEpisodeDataset
from src.variables import JESTER_OUTPUT_DIR
from src.utility import build_two_hand_adjacency
import json

def prepare_mapping(base_dir, exclude_none=True, min_samples=0):
    paths = glob(os.path.join(base_dir, 'Train', '*.npy')) + \
            glob(os.path.join(base_dir, 'Validation', '*.npy'))
    mapping = {}
    for p in paths:
        label = os.path.basename(p).split('_',1)[1].rsplit('.npy',1)[0]
        if exclude_none and label == 'None':
            continue
        mapping.setdefault(label, []).append(p)
    return {cls: ps for cls, ps in mapping.items() if len(ps) >= min_samples}

# Settings
N, K, Q, EPOCHS = 5,5,5,5
TRAIN_EPISODES, TEST_EPISODES = 100, 50

class_splits_file = 'class_splits.json'
BATCH_SIZE = 32

# ─── 0) 클래스 풀 분할 ─────────────────────────────────────────
mapping_all = prepare_mapping(JESTER_OUTPUT_DIR, min_samples=K+Q)   # Jester dataset

if not os.path.exists(class_splits_file):
    classes = list(mapping_all.keys())
    random.seed(42)
    random.shuffle(classes)
    n = len(classes)
    train_cls = classes[: int(0.6*n)]
    val_cls   = classes[int(0.6*n): int(0.8*n)]
    test_cls  = classes[int(0.8*n):]

    # 파일로 저장
    splits = {
        'train': train_cls,
        'val':   val_cls,
        'test':  test_cls
    }
    with open(class_splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
else:
    # 파일에서 불러오기
    with open(class_splits_file, 'r') as f:
        splits = json.load(f)
    train_cls = splits['train']
    val_cls   = splits['val']
    test_cls  = splits['test']

mapping_train = {c: mapping_all[c] for c in train_cls}
mapping_val   = {c: mapping_all[c] for c in val_cls}
mapping_test  = {c: mapping_all[c] for c in test_cls}
    
# # ─── 1) 백본 사전학습 (Pre-training) ────────────────────────────
# def load_npy(path, label):
#     # 1) EagerTensor → bytes (tf.string) → 파이썬 bytes
#     path_str = path.numpy().decode('utf-8')
#     # 2) .npy 로드
#     x = np.load(path_str)       # 이제 올바른 str 경로
#     x = x.astype('float32')
#     # 3) label도 numpy 스칼라로
#     y = np.int32(label.numpy())
#     return x, y

# def parse_npy(path, label):
#     x, y = tf.py_function(load_npy, [path, label], [tf.float32, tf.int32])
#     # shape 정보 복원
#     x = tf.ensure_shape(x, [37, 42, 3])
#     y = tf.ensure_shape(y, [])
#     # (T, V, C) → (C, V, T)
#     x = tf.transpose(x, [2,1,0])
#     return x, y

# # 1.1) tf.data 파이프라인
# class_to_idx = {c:i for i,c in enumerate(train_cls)}
# paths, labels = [], []
# for c, ps in mapping_train.items():
#     paths += ps
#     labels += [class_to_idx[c]] * len(ps)

# ds_pre = tf.data.Dataset.from_tensor_slices((paths, labels))
# ds_pre = ds_pre.shuffle(1000, seed=42) \
#             .map(parse_npy, num_parallel_calls=tf.data.AUTOTUNE) \
#             .batch(BATCH_SIZE) \
#             .prefetch(tf.data.AUTOTUNE)

# # 1.2) 모델 정의
# V = 42  # 노드 수
# A = build_two_hand_adjacency()  
# backbone = STGCN(in_channels=3, num_class=len(train_cls), A=A, num_layers=9)

# model_pre = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(3, 42, 37)),  # ← 이제 T=37, V=42, C=3이 명시됨
#     tf.keras.layers.Lambda(lambda x: tf.transpose(x, [0,3,2,1])),  # → (B, C, V, T)
#     backbone,                                       # → (B, len(train_cls))
#     tf.keras.layers.Activation('softmax')
# ])

# model_pre.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-3),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # 1.3) 학습
# total_samples = len(paths)
# val_size = int(0.1 * total_samples)

# # shuffle → 분리 → batch & prefetch
# ds_shuffled = ds_pre.unbatch() \
#                     .shuffle(buffer_size=total_samples, seed=42)  # unbatch 해서 shuffle 전체에 적용
# ds_val   = ds_shuffled.take(val_size) \
#                         .batch(BATCH_SIZE) \
#                         .prefetch(tf.data.AUTOTUNE)
# ds_train = ds_shuffled.skip(val_size) \
#                         .batch(BATCH_SIZE) \
#                         .prefetch(tf.data.AUTOTUNE)

# # 4) fit 에는 train/val 따로 지정
# model_pre.fit(
#     ds_train,
#     epochs=EPOCHS,
#     validation_data=ds_val
# )

# # 1.4) 사전학습된 백본 가중치 저장
# backbone.save_weights('models/backbone_pretrained.weights.h5')
# print("Pre-training completed and weights saved.")    

# ── 2) MAML/Meta-training 스크립트 초기화 ─────────────────
V = 42
A = build_two_hand_adjacency()  # 두 손의 adjacency matrix

# 2.1) ST-GCN 백본 인스턴스 생성
backbone = STGCN(in_channels=3, num_class=len(train_cls), A=A, num_layers=9)

# 2.2) 저장된 가중치 불러오기
backbone.load_weights('models/backbone_pretrained.weights.h5')
print("Loaded pretrained backbone weights for meta-training.")

# feature_extractor: ST-GCN 블록 + global_pool
feature_extractor = tf.keras.Sequential(
    backbone.blocks + [backbone.global_pool],
    name="feature_extractor"
)
feature_extractor.trainable = False              # 백본 완전 freeze


# ── 2) MAML 헤드 정의 ───────────────────────────────────────
class MAMLHead(tf.Module):
    def __init__(self, fe, emb_dim, way):
        super().__init__()
        self.fe = fe
        # head 파라미터만 inner/outer 양쪽에서 업데이트
        self.w = tf.Variable(tf.random.normal([emb_dim, way]), name='w')
        self.b = tf.Variable(tf.zeros([way]), name='b')

    def __call__(self, x, params=None):
        # x: (B, T, V, C) → (B, C, V, T)
        x = tf.transpose(x, [0,3,2,1])
        feat = self.fe(x, training=False)   # (B, emb_dim)
        w, b = params if params is not None else (self.w, self.b)
        return tf.matmul(feat, w) + b       # logits (B, way)

def sparse_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# ── 3) MAML 스텝: 헤드만 inner/outer 업데이트 ───────────────────
@tf.function
def maml_step(maml: MAMLHead, opt, sx, sy, qx, qy, inner_steps=1, alpha=1e-2):
    with tf.GradientTape() as outer_tape:
        fw, fb = maml.w, maml.b
        # inner-loop: support → fast weights
        for _ in range(inner_steps):
            with tf.GradientTape() as t1:
                t1.watch([fw, fb])
                ls = sparse_loss(maml(sx, params=[fw, fb]), sy)
            g1 = t1.gradient(ls, [fw, fb])
            fw, fb = fw - alpha*g1[0], fb - alpha*g1[1]

        # query loss with adapted head
        lq = sparse_loss(maml(qx, params=[fw, fb]), qy)

    # outer-loop: 헤드 파라미터만
    meta_grads = outer_tape.gradient(lq, [maml.w, maml.b])
    opt.apply_gradients(zip(meta_grads, [maml.w, maml.b]))
    return ls, lq

# ── 4) 메타-학습 + 메타-검증 루프 ─────────────────────────────────
def maml_meta_train(backbone, emb_dim, train_ds, val_ds, inner_lr, outer_lr, epochs):
    optimizer = tf.keras.optimizers.Adam(outer_lr)
    # MAML 헤드 초기화
    maml = MAMLHead(feature_extractor, emb_dim, way=len(train_cls))
    _ = backbone(tf.zeros((1,3,42,37)))  # build
    inner_steps = 1  # inner-loop step 수
    for epoch in range(1, epochs+1):
        tot_lq, tot_acc, cnt = 0.0, 0.0, 0
        for sx, sy, qx, qy in train_ds:
            # Inner-loop: support set으로 fast weights 업데이트
            ls, lq = maml_step(maml, optimizer, sx, sy, qx, qy, inner_steps=inner_steps, alpha=inner_lr)
            tot_lq += lq.numpy()

            # adapted head로 query accuracy
            fw, fb = maml.w, maml.b
            for _ in range(1):
                with tf.GradientTape() as t2:
                    t2.watch([fw, fb])
                    lsi = sparse_loss(maml(sx, params=[fw, fb]), sy)
                g2 = t2.gradient(lsi, [fw, fb])
                fw, fb = fw - inner_lr*g2[0], fb - inner_lr*g2[1]   
            preds_q = tf.argmax(maml(qx, params=[fw, fb]), axis=1, output_type=sy.dtype)
            tot_acc += tf.reduce_mean(tf.cast(preds_q==qy, tf.float32)).numpy()
            cnt += 1
        print(f"[Epoch {epoch}] Meta-loss: {tot_lq/cnt:.4f}, Meta-acc: {tot_acc/cnt:.4f}")
        # ── 메타-검증 ─────────────────────────────────────────────
        val_loss, val_acc, val_cnt = 0.0, 0.0, 0
        for vsx, vsy, vqx, vqy in val_ds:
            # Inner-loop: support set으로 fast weights 업데이트
            ls, lq = maml_step(maml, optimizer, vsx, vsy, vqx, vqy, inner_steps=inner_steps, alpha=inner_lr)
            val_loss += lq.numpy()

            # adapted head로 query accuracy
            fw, fb = maml.w, maml.b
            for _ in range(1):
                with tf.GradientTape() as t2:
                    t2.watch([fw, fb])
                    lsi = sparse_loss(maml(vsx, params=[fw, fb]), vsy)
                g2 = t2.gradient(lsi, [fw, fb])
                fw, fb = fw - inner_lr*g2[0], fb - inner_lr*g2[1]
            preds_vq = tf.argmax(maml(vqx, params=[fw, fb]), axis=1, output_type=vsy.dtype)
            val_acc += tf.reduce_mean(tf.cast(preds_vq==vqy, tf.float32)).numpy()
            val_cnt += 1    
        print(f"[Validation] Meta-loss: {val_loss/val_cnt:.4f}, Meta-acc: {val_acc/val_cnt:.4f}")
    return maml


# def meta_train(meta_ds, emb_dim=64, way=27, epochs=30, inner=1, alpha=1e-2):
    # maml = MAMLHead(feature_extractor, emb_dim, way)
    # opt  = tf.keras.optimizers.Adam(1e-3)
    # for e in range(1, epochs+1):
    #     tot_lq, tot_acc, cnt = 0.0, 0.0, 0
    #     for sx, sy, qx, qy in meta_ds:
    #         ls, lq = maml_step(maml, opt, sx, sy, qx, qy, inner, alpha)
    #         tot_lq += lq.numpy()
    #         # adapted head로 query accuracy
    #         fw, fb = maml.w, maml.b
    #         for _ in range(inner):
    #             with tf.GradientTape() as t2:
    #                 t2.watch([fw, fb])
    #                 lsi = sparse_loss(maml(sx, params=[fw, fb]), sy)
    #             g2 = t2.gradient(lsi, [fw, fb])
    #             fw, fb = fw - alpha*g2[0], fb - alpha*g2[1]
    #         preds_q = tf.argmax(maml(qx, params=[fw, fb]), axis=1, output_type=sy.dtype)
    #         tot_acc += tf.reduce_mean(tf.cast(preds_q==qy, tf.float32)).numpy()
    #         cnt += 1
    #     print(f"[Epoch {e}] Meta-loss: {tot_lq/cnt:.4f}, Meta-acc: {tot_acc/cnt:.4f}")
    # return maml

# # ── 3) MAML 루프 정의 ────────────────────────────────────
# def maml_meta_train(backbone, train_ds, val_ds, inner_lr=1e-2, outer_lr=1e-3, epochs=20):
#     # 0) build
#     _ = backbone(tf.zeros((1, 3, 42, 37)), training=False)
#     optimizer = tf.keras.optimizers.Adam(outer_lr)

#     for epoch in range(1, epochs+1):
#         # 메타-트레이닝 지표 초기화
#         meta_loss = 0.0
#         meta_acc  = 0.0
#         task_count = 0
#         # 1) 원본 θ₀ 백업
#         orig_vars = [tf.identity(v) for v in backbone.trainable_variables]

#         for sup_x, sup_y, qry_x, qry_y in train_ds:
#             task_count += 1

#             # 2) Inner-loop: support → θ′ 리스트
#             with tf.GradientTape() as tape1:
#                 logits_s = backbone(tf.transpose(sup_x, [0,3,2,1]), training=True)
#                 loss_s   = tf.reduce_mean(
#                     tf.keras.losses.sparse_categorical_crossentropy(
#                         sup_y, logits_s, from_logits=True))
            
#             # 2) θ₀에 대한 gradient  
#             grads_s = tape1.gradient(loss_s, backbone.trainable_variables)

#             # 3) theta_prime 리스트 생성 (Tensor 리스트)
#             theta_prime = [
#                 v - inner_lr * g
#                 for v, g in zip(backbone.trainable_variables, grads_s)
#             ]

#             # 4) θ′ 적용
#             for v, new_v in zip(backbone.trainable_variables, theta_prime):
#                 v.assign(new_v)

#             # 5) Outer-loop: query 손실·정확도 + meta-gradient
#             with tf.GradientTape() as tape2:
#                 logits_q = backbone(tf.transpose(qry_x, [0,3,2,1]), training=True)
#                 loss_q   = tf.reduce_mean(
#                     tf.keras.losses.sparse_categorical_crossentropy(
#                         qry_y, logits_q, from_logits=True))
                
#                 # query 정확도 계산
#                 preds_q = tf.argmax(logits_q, axis=1, output_type=qry_y.dtype)
#                 acc_q   = tf.reduce_mean(tf.cast(tf.equal(preds_q, qry_y), tf.float32))

            
#             # 6) θ′에 대한 gradient 
#             grads_q = tape2.gradient(loss_q, backbone.trainable_variables)
#             grad_q_norm = tf.linalg.global_norm(grads_q)
#             # print(f"  Episode {task_count}: sup_loss={loss_s.numpy():.4f}, qry_loss={loss_q.numpy():.4f}, grad_q_norm={grad_q_norm:.4f}")
            
#             # 메타 지표 누적
#             meta_loss += loss_q.numpy()
#             meta_acc  += acc_q.numpy()

#             # 7) θ₀ 복원
#             for v, orig in zip(backbone.trainable_variables, orig_vars):
#                 v.assign(orig)

#             # 8) meta-gradients 누적
#             if task_count == 1:
#                 meta_grads = [tf.zeros_like(v) for v in backbone.trainable_variables]
#             meta_grads = [mg + gq for mg, gq in zip(meta_grads, grads_q)]

#         # 9) meta-grads 평균 & Outer update
#         meta_grads = [mg / tf.cast(task_count, tf.float32) for mg in meta_grads]
#         meta_norm = tf.linalg.global_norm(meta_grads)
#         print(f" Meta-grad norm before apply: {meta_norm:.4f}")
#         optimizer.apply_gradients(zip(meta_grads, backbone.trainable_variables))

#         # 에폭당 메타-트레이닝 지표 출력
#         print(f"[Epoch {epoch}] Meta-train  Loss={meta_loss/task_count:.4f}  Acc={meta_acc/task_count:.4f}")

#         # ── 메타-검증 진행 ───────────────────────────
#         if val_ds is not None:
#             val_loss = 0.0
#             val_acc  = 0.0
#             val_count = 0

#             # 0) θ₀ 백업 (메타-트레이닝 루프에서 이미 백업해 둔 orig_vars 사용)
#             orig_vars = [tf.identity(v) for v in backbone.trainable_variables]

#             for vsup_x, vsup_y, vqry_x, vqry_y in val_ds:
#                 val_count += 1

#                 # 1) Inner-loop: support → θ′ 리스트
#                 with tf.GradientTape() as t1:
#                     logits_s = backbone(tf.transpose(vsup_x, [0,3,2,1]), training=False)
#                     loss_s   = tf.reduce_mean(
#                         tf.keras.losses.sparse_categorical_crossentropy(
#                             vsup_y, logits_s, from_logits=True))
#                 grads_s = t1.gradient(loss_s, backbone.trainable_variables)
#                 theta_prime = [
#                     v - inner_lr * g
#                     for v, g in zip(backbone.trainable_variables, grads_s)
#                 ]

#                 # 2) θ′ 적용
#                 for v, new_v in zip(backbone.trainable_variables, theta_prime):
#                     v.assign(new_v)

#                 # 3) Query 평가
#                 logits_q = backbone(tf.transpose(vqry_x, [0,3,2,1]), training=False)
#                 loss_q   = tf.reduce_mean(
#                     tf.keras.losses.sparse_categorical_crossentropy(
#                         vqry_y, logits_q, from_logits=True))
#                 preds_q  = tf.argmax(logits_q, axis=1, output_type=vqry_y.dtype)
#                 acc_q    = tf.reduce_mean(tf.cast(tf.equal(preds_q, vqry_y), tf.float32))

#                 val_loss += loss_q.numpy()
#                 val_acc  += acc_q.numpy()

#                 # 4) θ₀ 복원
#                 for v, orig in zip(backbone.trainable_variables, orig_vars):
#                     v.assign(orig)

#             print(f"           Meta-val    Loss={val_loss/val_count:.4f}  Acc={val_acc/val_count:.4f}")


# # ── 4) Few-shot 평가 함수 정의 ─────────────────────────────
# def evaluate_few_shot(backbone, test_ds, inner_lr=1e-2, episodes=100):
#     """
#     backbone: 메타학습된 STGCN 모델
#     test_ds:  GestureEpisodeDataset(...).get_tf_dataset()로 생성된 tf.data.Dataset
#     inner_lr: inner-loop 학습률
#     episodes: 에피소드 수 (== len(test_ds))
#     """
#     total_acc = 0.0

#     # θ₀ 백업
#     orig_vars = [tf.identity(v) for v in backbone.trainable_variables]

#     for episode_idx, (sup_x, sup_y, qry_x, qry_y) in enumerate(test_ds, start=1):
#         # 1) Inner-loop: support → θ′ 리스트 계산
#         with tf.GradientTape() as tape:
#             logits_s = backbone(tf.transpose(sup_x, [0,3,2,1]), training=False)
#             loss_s   = tf.reduce_mean(
#                 tf.keras.losses.sparse_categorical_crossentropy(
#                     sup_y, logits_s, from_logits=True))
#         grads_s = tape.gradient(loss_s, backbone.trainable_variables)
#         theta_prime = [
#             v - inner_lr * g
#             for v, g in zip(backbone.trainable_variables, grads_s)
#         ]

#         # 2) θ′ 한 번에 적용
#         for v, new_v in zip(backbone.trainable_variables, theta_prime):
#             v.assign(new_v)

#         # 3) Query 평가
#         logits_q = backbone(tf.transpose(qry_x, [0,3,2,1]), training=False)
#         preds_q  = tf.argmax(logits_q, axis=1, output_type=qry_y.dtype)
#         acc_q    = tf.reduce_mean(tf.cast(tf.equal(preds_q, qry_y), tf.float32))
#         total_acc += acc_q.numpy()

#         # 4) θ₀ 복원
#         for v, orig in zip(backbone.trainable_variables, orig_vars):
#             v.assign(orig)

#     avg_acc = total_acc / episodes
#     print(f"*** Test Accuracy over {episodes} episodes: {avg_acc:.4f} ***")

# ── 4) Few-shot 평가 함수 정의 ─────────────────────────────
def evaluate_few_shot_mamlhead(maml: MAMLHead, test_ds, inner_lr, inner_steps=1):
    total_acc = 0.0
    count = 0

    for sx, sy, qx, qy in test_ds:
        # 1) fast weights 초기화
        fw, fb = maml.w, maml.b

        # 2) inner-loop로 support set에 맞춰 헤드 어댑테이션
        for _ in range(inner_steps):
            with tf.GradientTape() as tape:
                tape.watch([fw, fb])
                logits_s = maml(sx, params=[fw, fb])
                loss_s = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy, logits=logits_s))
            g1 = tape.gradient(loss_s, [fw, fb])
            fw, fb = fw - inner_lr * g1[0], fb - inner_lr * g1[1]

        # 3) adapted head로 query 평가
        logits_q = maml(qx, params=[fw, fb])
        preds_q = tf.argmax(logits_q, axis=1, output_type= qy.dtype)
        acc_q = tf.reduce_mean(tf.cast(preds_q == qy, tf.float32)).numpy()

        total_acc += acc_q
        count += 1

    print(f"*** Meta-test Few-Shot Accuracy: {total_acc/count:.4f} ***")

        
# ── 5) 데이터 로더 준비 ────────────────────────────────────
train_ds = GestureEpisodeDataset(mapping_train, N, K, Q, TRAIN_EPISODES).get_tf_dataset()
val_ds   = GestureEpisodeDataset(mapping_val,   N, K, Q, TEST_EPISODES).get_tf_dataset()
test_ds  = GestureEpisodeDataset(mapping_test,  N, K, Q, TEST_EPISODES).get_tf_dataset()
# train_ds = train_ds.repeat()  
# val_ds   = val_ds.repeat()    
# test_ds  = test_ds.repeat()   

# ── 6) 실행 ─────────────────────────────────────────────
if __name__ == '__main__':
    # 0) backbone build
    _ = backbone(tf.zeros((1,3,42,37)))

    # 1) emb_dim 계산
    feature_extractor.build((None,3,42,37))
    emb_dim = feature_extractor.output_shape[-1]
    print("Embedding dim:", emb_dim)

    # 2) 메타학습
    maml = maml_meta_train(
        backbone,
        emb_dim,
        train_ds,
        val_ds,
        1e-2,
        1e-3,
        EPOCHS
    )

    # 3) 가중치 저장/로드
    backbone.save_weights('models/backbone_maml_trained.weights.h5')
    backbone.load_weights('models/backbone_maml_trained.weights.h5')

    # 4) Few-shot 평가
    print("\nRunning few-shot evaluation on novel classes:")
    evaluate_few_shot_mamlhead(
        maml,
        test_ds,
        1e-2,
        TEST_EPISODES
    )

    

#     # 1) 메타학습
#     maml_meta_train(backbone, train_ds, val_ds, inner_lr=1e-2, outer_lr=1e-3, epochs=EPOCHS)
#     # 2) 저장
#     backbone.save_weights('models/backbone_maml_trained.weights.h5')
#     print("Meta-training completed and weights saved.")
#     # 3) 평가를 위해 다시 로드 (필요 시)
#     backbone.load_weights('models/backbone_maml_trained.weights.h5')
#     # 4) Few-shot 평가
#     print("\nRunning few-shot evaluation on novel classes:")
#     evaluate_few_shot(backbone, test_ds, inner_lr=1e-2, episodes=TEST_EPISODES)

    
    
