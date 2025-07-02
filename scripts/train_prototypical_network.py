import os
import random
import numpy as np
import tensorflow as tf
from glob import glob

from src.data_loader import GestureEpisodeDataset
from src.variables import JESTER_OUTPUT_DIR, CONGD_OUTPUT_DIR
from src.st_gcn import STGCN
from src.utility import build_two_hand_adjacency, prepare_mapping_jester
from src.embedding_utils import extract_embeddings_cnn, save_embeddings_to_npy

import json

class CNNEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, x, training=False):
        return self.model(x)

class ProtoNetCNN(tf.keras.Model):
    def __init__(self, N, K, Q, embedding_dim=128):
        super().__init__()
        self.N, self.K, self.Q = N, K, Q
        self.encoder = CNNEncoder(embedding_dim)

    def call(self, support, query, training=False):
        def preprocess(x):
            # (B, T, V, C) → (B, V, C*T, 1)
            B, T, V, C = x.shape
            x = tf.reshape(x, [B, V, C*T])
            x = tf.expand_dims(x, -1)  # CNN input: (B, H, W, 1)
            return x

        sup = preprocess(support)
        qry = preprocess(query)

        sup_emb = self.encoder(sup, training=training)
        qry_emb = self.encoder(qry, training=training)

        sup_emb = tf.reshape(sup_emb, (self.N, self.K, -1))
        prototypes = tf.reduce_mean(sup_emb, axis=1)

        a = tf.expand_dims(qry_emb, 1)
        b = tf.expand_dims(prototypes, 0)
        dists = tf.reduce_sum((a - b)**2, axis=2)
        return tf.nn.log_softmax(-dists, axis=1)


class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, N, K, Q, embedding_dim=128, num_layers=4):
        super().__init__()
        self.N, self.K, self.Q = N, K, Q
        A = build_two_hand_adjacency()
        self.backbone = STGCN(
            in_channels=3,
            num_class=embedding_dim,
            A=A,
            num_layers=num_layers
        )

    def call(self, support, query, training=False):
        # support: (N*K, T, V, C), query: (N*Q, T, V, C)
        sup = tf.transpose(support, [0,3,2,1])  # (N*K, C, V, T)
        qry = tf.transpose(query,   [0,3,2,1])  # (N*Q, C, V, T)
        sup_emb = self.backbone(sup, training=training)
        qry_emb = self.backbone(qry, training=training)
        sup_emb = tf.reshape(sup_emb, (self.N, self.K, -1))
        prototypes = tf.reduce_mean(sup_emb, axis=1)  # (N, D)

        # Euclidean distance
        a = tf.expand_dims(qry_emb, 1)
        b = tf.expand_dims(prototypes, 0)
        dists = tf.reduce_sum((a - b) ** 2, axis=2)
        return tf.nn.log_softmax(-dists, axis=1)

# def train(model, dataset, optimizer, epochs=20):
#     for epoch in range(1, epochs+1):
#         total_loss = total_acc = 0.0
#         step = 0
#         for support_x, support_y, query_x, query_y in dataset:
#             with tf.GradientTape() as tape:
#                 log_p = model(support_x, query_x, training=True)
#                 idx = tf.stack([tf.range(model.N * model.Q), query_y], axis=1)
#                 loss = -tf.reduce_mean(tf.gather_nd(log_p, idx))
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#             preds = tf.argmax(log_p, axis=1, output_type=tf.int32)
#             acc = tf.reduce_mean(tf.cast(preds == query_y, tf.float32))
#             total_loss += loss.numpy()
#             total_acc += acc.numpy()
#             step += 1
#         print(f"Epoch {epoch}: Loss={total_loss/step:.4f}, Acc={total_acc/step:.4f}")


def train(model, train_ds, val_ds, optimizer, epochs=20):
    for epoch in range(1, epochs+1):
        # ——— Training ———
        total_loss = total_acc = 0.0
        steps = 0
        for support_x, support_y, query_x, query_y in train_ds:
            with tf.GradientTape() as tape:
                log_p = model(support_x, query_x, training=True)
                idx = tf.stack([tf.range(model.N * model.Q), query_y], axis=1)
                loss = -tf.reduce_mean(tf.gather_nd(log_p, idx))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            preds = tf.argmax(log_p, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(preds == query_y, tf.float32))

            total_loss += loss.numpy()
            total_acc  += acc.numpy()
            steps += 1

        train_loss = total_loss / steps
        train_acc  = total_acc  / steps

        # ——— Validation ———
        val_loss = val_acc = 0.0
        val_steps = 0
        for support_x, support_y, query_x, query_y in val_ds:
            log_p = model(support_x, query_x, training=False)
            idx = tf.stack([tf.range(model.N * model.Q), query_y], axis=1)
            loss = -tf.reduce_mean(tf.gather_nd(log_p, idx))

            preds = tf.argmax(log_p, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(preds == query_y, tf.float32))

            val_loss += loss.numpy()
            val_acc  += acc.numpy()
            val_steps += 1

        val_loss /= val_steps
        val_acc  /= val_steps

        print(f"Epoch {epoch:02d}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}  |  "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


def evaluate(model, dataset):
    total_acc = 0.0
    step = 0
    for supp_x, supp_y, qry_x, qry_y in dataset:
        log_p = model(supp_x, qry_x, training=False)
        preds = tf.argmax(log_p, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(preds == qry_y, tf.float32))
        total_acc += acc.numpy()
        step += 1
    print(f"Meta-test Accuracy: {total_acc/step:.4f}")

def main():
    # Settings
    N, K, Q = 5, 1, 1
    EMB_DIM = 64
    TRAIN_EPISODES, TEST_EPISODES = 2000, 500
    # TRAIN_EPISODES, TEST_EPISODES = 100, 50
    random.seed(42)

    # # Prepare data
    # mapping = prepare_mapping_jester(JESTER_OUTPUT_DIR, min_samples=K+Q)
    # classes = list(mapping.keys())
    # random.shuffle(classes)
    # split = int(0.6 * len(classes))
    # train_map = {c: mapping[c] for c in classes[:split]}
    # test_map  = {c: mapping[c] for c in classes[split:]}

    # train_ds = GestureEpisodeDataset(train_map, N, K, Q, TRAIN_EPISODES).get_tf_dataset()
    # test_ds  = GestureEpisodeDataset(test_map,  N, K, Q, TEST_EPISODES).get_tf_dataset()

    # ─── 0) 클래스 풀 분할 ─────────────────────────────────────────
    class_splits_file = 'class_splits.json'
    mapping_all = prepare_mapping_jester(JESTER_OUTPUT_DIR, min_samples=K+Q)   # Jester dataset

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

    train_ds = GestureEpisodeDataset(mapping_train, N, K, Q, TRAIN_EPISODES).get_tf_dataset()
    val_ds   = GestureEpisodeDataset(mapping_val,   N, K, Q, TEST_EPISODES).get_tf_dataset()
    test_ds  = GestureEpisodeDataset(mapping_test,  N, K, Q, TEST_EPISODES).get_tf_dataset()

    model_cnn = ProtoNetCNN(N, K, Q, embedding_dim=EMB_DIM)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    train(model_cnn, train_ds, val_ds, optimizer, epochs=20)
    evaluate(model_cnn, test_ds)
    embeds, labels = extract_embeddings_cnn(model_cnn, test_ds)
    save_embeddings_to_npy(embeds, labels, prefix='cnn', out_dir='umap_outputs')


    # # Model and optimizer
    # model = PrototypicalNetwork(N, K, Q, EMB_DIM)
    # optimizer = tf.keras.optimizers.Adam(1e-3)

    # # Training and evaluation
    # train(model, train_ds, val_ds, optimizer, epochs=20)
    # evaluate(model, test_ds)


if __name__ == '__main__':
    main()


# import os
# import numpy as np
# import tensorflow as tf
# import random
# from src.data_loader import GestureEpisodeDataset
# from src.variables import *  # FRAMES, etc.
# from src.st_gcn import STGCN  # 위에서 만든 stgcn_tf.py

# # --- Prototypical Network ---
# class PrototypicalNetwork(tf.keras.Model):
#     def __init__(self, N, K, Q, embedding_dim=128):
#         super().__init__()
#         self.N, self.K, self.Q = N, K, Q
#         # self.encoder = tf.keras.Sequential([
#         #     tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
#         #     tf.keras.layers.MaxPool2D(),
#         #     tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#         #     tf.keras.layers.GlobalAveragePooling2D(),
#         #     tf.keras.layers.Dense(embedding_dim)
#         # ])

#         # 3) 모델 생성 및 컴파일
#         # 3.1) MediaPipe Hands 21개 랜드마크 인덱스에 따른 간선(edge) 목록
#         edges = [
#             (0,1),(1,2),(2,3),(3,4),         # Thumb
#             (0,5),(5,6),(6,7),(7,8),         # Index
#             (0,9),(9,10),(10,11),(11,12),    # Middle
#             (0,13),(13,14),(14,15),(15,16),  # Ring
#             (0,17),(17,18),(18,19),(19,20)   # Pinky
#         ]
#         V_hand = 21

#         # 3.2) 한 손 adjacency 초기화
#         A_hand = np.zeros((V_hand, V_hand), dtype=np.float32)
#         for i, j in edges:
#             A_hand[i, j] = 1
#             A_hand[j, i] = 1
#         # self-loop 추가
#         np.fill_diagonal(A_hand, 1)

#         # 3.3) 정규화
#         D = np.diag(1.0 / np.sqrt(A_hand.sum(axis=1)))
#         A_hand = D @ A_hand @ D

#         # 3.4) 두 손 합치기 (42×42 block-diagonal)
#         V = V_hand * 2
#         A = np.zeros((V, V), dtype=np.float32)
#         A[:V_hand, :V_hand] = A_hand      # 왼손 블록
#         A[V_hand:, V_hand:] = A_hand      # 오른손 블록

#         # 3.5) 양손 간 edge 추가 (ex: 손목과 손끝 연결)
#         # 손목 (0 ↔ 21)
#         A[0, 21] = A[21, 0] = 1

#         # 양손 손가락 끝끼리 연결 (Thumb~Pinky)
#         left_tips = [4, 8, 12, 16, 20]
#         right_tips = [25, 29, 33, 37, 41]
#         for l, r in zip(left_tips, right_tips):
#             A[l, r] = A[r, l] = 1

#         # 3.6) 전체 정규화
#         D = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
#         A = D @ A @ D

#         # # 2) ST-GCN 백본 생성: num_class은 분류 헤드 대신 feature extractor로만 쓸 거라  None 또는 임시값
#         self.backbone = STGCN(
#             in_channels=3,
#             num_class=embedding_dim,  # 임베딩 크기로 설정해도 되고
#             A=A,
#             num_layers=4
#         )
#         # # 3) global pooling + 추가 projection (선택)
#         # self.pool = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')
#         # self.fc   = tf.keras.layers.Dense(embedding_dim)

#     def call(self, support, query, training=False):
#         # support: (N*K, T, V, C)
#         # query:   (N*Q, T, V, C)
        
#         # 1) 채널·버텍스·타임 순서로 변경
#         #    → STGCN이 기대하는 (batch, C, V, T)
#         sup = tf.transpose(support, [0, 3, 2, 1])  # (N*K, C, V, T)
#         qry = tf.transpose(query,   [0, 3, 2, 1])  # (N*Q, C, V, T)

#         # 2) ST-GCN 백본 통과
#         sup_emb = self.backbone(sup, training=training)  # (N*K, D_emb, V, T')
#         qry_emb = self.backbone(qry, training=training)  # (N*Q, D_emb, V, T')

#         # 4) 프로토타입 계산
#         #    - support 임베딩을 (N, K, D)로 reshape
#         sup_emb = tf.reshape(sup_emb, (self.N, self.K, -1))
#         prototypes = tf.reduce_mean(sup_emb, axis=1)     # (N, D)


#         # 5) query 임베딩과 프로토타입 간 거리 계산
#         #    - 유클리드 거리
#         def euclid(a, b):
#             # a: (N*Q, D), b: (N, D)
#             a = tf.expand_dims(a, 1)  # (N*Q, 1, D)
#             b = tf.expand_dims(b, 0)  # (1, N, D)
#             return tf.reduce_sum((a - b)**2, axis=2)  # (N*Q, N)

#         dists = euclid(qry_emb, prototypes)  # (N*Q, N)

#         # 6) negative distance에 softmax → log-prob
#         return tf.nn.log_softmax(-dists, axis=1)
    
#     # def call(self, support, query, training=False):
#     #     def reshape_x(x):
#     #         b, T, V, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
#     #         x_ = tf.transpose(x, [0,2,3,1])
#     #         return tf.reshape(x_, (b, V, C * T, 1))

#     #     # support/query: (N*K, T, V, C)
#     #     sup = tf.transpose(support, [0, 3, 2, 1])  # → (N*K, C, V, T)
#     #     qry = tf.transpose(query,  [0, 3, 2, 1])    # → (N*Q, C, V, T)

#     #     # ST-GCN backbone 통과
#     #     sup_feat = self.backbone(sup, training=training)  # (N*K, F, V', T') 또는 (N*K, T', F) 형태
#     #     qry_feat = self.backbone(qry, training=training)

#     #      # 마지막 차원 pooling & 임베딩
#     #     sup_emb = self.fc(tf.reduce_mean(sup_feat, axis=[2,3]))  # (N*K, D)
#     #     qry_emb = self.fc(tf.reduce_mean(qry_feat, axis=[2,3]))

#     #     # sup = reshape_x(support)
#     #     # qry = reshape_x(query)
#     #     # sup_emb = self.encoder(sup,  training=training)  # (N*K, D)
#     #     # qry_emb = self.encoder(qry,  training=training)  # (N*Q, D)

#     #     # sup_emb = tf.reshape(sup_emb, (N, K, -1))
#     #     # prototypes = tf.reduce_mean(sup_emb, axis=1)  # (N, D)

#     #     def euclidean_dist(a, b):
#     #         a_exp = tf.expand_dims(a, 1)
#     #         b_exp = tf.expand_dims(b, 0)
#     #         return tf.reduce_sum(tf.square(a_exp - b_exp), axis=2)

#     #     # dists = euclidean_dist(qry_emb, prototypes)  # (N*Q, N)
#     #     dists = euclidean_dist(qry_emb, sup_emb)  # (N*Q, N*K)
#     #     return tf.nn.log_softmax(-dists, axis=1)

# # --- Training Loop ---
# def train_prototypical(model, dataset, optimizer, epochs=20):
#     N, Q = model.N, model.Q
#     for epoch in range(1, epochs + 1):
#         total_loss, total_acc = 0.0, 0.0
#         count = 0
#         for support_x, support_y, query_x, query_y in dataset:
#             count += 1
#             with tf.GradientTape() as tape:
#                 log_p_y = model(support_x, query_x, training=True)
#                 # model.N, model.Q로부터 인덱스 계산
#                 indices = tf.stack([tf.range(N * Q), query_y], axis=1)
#                 loss = -tf.reduce_mean(tf.gather_nd(log_p_y, indices))

#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))

#             preds = tf.argmax(log_p_y, axis=1, output_type=tf.int32)
#             acc = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))

#             total_loss += loss.numpy()
#             total_acc  += acc.numpy()

#         print(f"Epoch {epoch}: Loss={total_loss/count:.4f}, Acc={total_acc/count:.4f}")

# # --- Evaluation Function ---
# def eval_prototypical(model, dataset):
#     """
#     model: PrototypicalNetwork 인스턴스 (model.N, model.Q 사용)
#     dataset: GestureEpisodeDataset(...).get_tf_dataset() 로 생성한 평가용 tf.data.Dataset
#     """
#     total_acc, count = 0.0, 0
#     N, Q = model.N, model.Q

#     for support_x, support_y, query_x, query_y in dataset:
#         # 추론 모드
#         log_p_y = model(support_x, query_x, training=False)  # (N*Q, N)
#         preds = tf.argmax(log_p_y, axis=1, output_type=tf.int32)
#         acc   = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))

#         total_acc += acc.numpy()
#         count += 1

# # --- Main Script ---
# if __name__ == '__main__':
#     from glob import glob

#     # # conGD
#     # all_paths = glob(CONGD_OUTPUT_DIR + '/*.npy')
#     # mapping = {}
#     # for p in all_paths:
#     #     label = os.path.basename(p).split('_', 5)[-1][3:].rsplit('.npy', 1)[0]  # "lblXXX"
#     #     mapping.setdefault(label, []).append(p)

#     # jester
#     base_dir = JESTER_OUTPUT_DIR
#     train_paths = glob(os.path.join(base_dir, 'Train', '*.npy'))
#     test_paths  = glob(os.path.join(base_dir, 'Validation', '*.npy'))
#     all_paths = train_paths + test_paths
#     print(f"Found {len(all_paths)} total gesture paths.")

#     # build label->paths
#     mapping = {}
#     for p in all_paths:
#         label = os.path.basename(p).split('_', 1)[1].rsplit('.npy',1)[0]
#         if label == 'None': continue
#         mapping.setdefault(label, []).append(p)

#     # filter classes and ensure enough for episode
#     N, K, Q = 10, 2, 2
#     mapping = {cls: ps for cls, ps in mapping.items() if len(ps) >= (K+Q)}   

#     # # dataset
#     # episodes = 2000
#     # epi_ds = GestureEpisodeDataset(mapping, N, K, Q, episodes).get_tf_dataset()

#     # model & optimizer
#     model = PrototypicalNetwork(N=N, K=K, Q=Q, embedding_dim=64)
#     optimizer = tf.keras.optimizers.Adam(1e-3)

#     # 클래스 리스트
#     classes = sorted(mapping.keys())

#     # 예시: 27개 중 랜덤으로 20개를 train pool, 나머지를 test pool
#     import random
#     random.seed(42)
#     train_classes = random.sample(classes, 17)
#     test_classes = [c for c in classes if c not in train_classes]

#     # train/test pool dict 생성
#     train_mapping = {c: mapping[c] for c in train_classes}
#     test_mapping  = {c: mapping[c] for c in test_classes}

#     # filter classes and ensure enough for episode
#     N, K, Q = 10, 2, 2
#     # mapping = {cls: ps for cls, ps in mapping.items() if len(ps) >= (K+Q)}   

#     # dataset
#     # train_ds: 2000 에피소드
#     episode = 2000
#     train_ds = GestureEpisodeDataset(
#             train_mapping,  # class_to_paths 인자
#             N,              # N
#             K,              # K
#             Q,              # Q
#             episode        # episodes_per_epoch
#         ).get_tf_dataset()

#     # test_ds: 500 에피소드 (예시)
#     episode = 500
#     test_ds = GestureEpisodeDataset(
#         test_mapping,
#         N, K, Q,
#         episode,
#     ).get_tf_dataset()

#     # model & optimizer
#     model = PrototypicalNetwork(N=N, K=K, Q=Q, embedding_dim=64)
#     optimizer = tf.keras.optimizers.Adam(1e-3)

#     # train
#     train_prototypical(model, train_ds, optimizer, epochs=20)

#     # evaluate
#     eval_prototypical(model, test_ds)



