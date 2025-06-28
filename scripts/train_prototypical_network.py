import os
import numpy as np
import tensorflow as tf
import random
from src.data_loader import GestureEpisodeDataset
from src.variables import *  # FRAMES, etc.
from src.st_gcn import STGCN  # 위에서 만든 stgcn_tf.py

# --- Prototypical Network ---
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, N, K, Q, embedding_dim=128):
        super().__init__()
        self.N, self.K, self.Q = N, K, Q
        # self.encoder = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        #     tf.keras.layers.MaxPool2D(),
        #     tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        #     tf.keras.layers.GlobalAveragePooling2D(),
        #     tf.keras.layers.Dense(embedding_dim)
        # ])
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

        # # 2) ST-GCN 백본 생성: num_class은 분류 헤드 대신 feature extractor로만 쓸 거라  None 또는 임시값
        self.backbone = STGCN(
            in_channels=3,
            num_class=embedding_dim,  # 임베딩 크기로 설정해도 되고
            A=A,
            num_layers=9
        )
        # # 3) global pooling + 추가 projection (선택)
        # self.pool = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')
        # self.fc   = tf.keras.layers.Dense(embedding_dim)

    def call(self, support, query, training=False):
        # support: (N*K, T, V, C)
        # query:   (N*Q, T, V, C)
        
        # 1) 채널·버텍스·타임 순서로 변경
        #    → STGCN이 기대하는 (batch, C, V, T)
        sup = tf.transpose(support, [0, 3, 2, 1])  # (N*K, C, V, T)
        qry = tf.transpose(query,   [0, 3, 2, 1])  # (N*Q, C, V, T)

        # 2) ST-GCN 백본 통과
        sup_emb = self.backbone(sup, training=training)  # (N*K, D_emb, V, T')
        qry_emb = self.backbone(qry, training=training)  # (N*Q, D_emb, V, T')

        # 4) 프로토타입 계산
        #    - support 임베딩을 (N, K, D)로 reshape
        sup_emb = tf.reshape(sup_emb, (self.N, self.K, -1))
        prototypes = tf.reduce_mean(sup_emb, axis=1)     # (N, D)


        # 5) query 임베딩과 프로토타입 간 거리 계산
        #    - 유클리드 거리
        def euclid(a, b):
            # a: (N*Q, D), b: (N, D)
            a = tf.expand_dims(a, 1)  # (N*Q, 1, D)
            b = tf.expand_dims(b, 0)  # (1, N, D)
            return tf.reduce_sum((a - b)**2, axis=2)  # (N*Q, N)

        dists = euclid(qry_emb, prototypes)  # (N*Q, N)

        # 6) negative distance에 softmax → log-prob
        return tf.nn.log_softmax(-dists, axis=1)
    
    # def call(self, support, query, training=False):
    #     def reshape_x(x):
    #         b, T, V, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    #         x_ = tf.transpose(x, [0,2,3,1])
    #         return tf.reshape(x_, (b, V, C * T, 1))

    #     # support/query: (N*K, T, V, C)
    #     sup = tf.transpose(support, [0, 3, 2, 1])  # → (N*K, C, V, T)
    #     qry = tf.transpose(query,  [0, 3, 2, 1])    # → (N*Q, C, V, T)

    #     # ST-GCN backbone 통과
    #     sup_feat = self.backbone(sup, training=training)  # (N*K, F, V', T') 또는 (N*K, T', F) 형태
    #     qry_feat = self.backbone(qry, training=training)

    #      # 마지막 차원 pooling & 임베딩
    #     sup_emb = self.fc(tf.reduce_mean(sup_feat, axis=[2,3]))  # (N*K, D)
    #     qry_emb = self.fc(tf.reduce_mean(qry_feat, axis=[2,3]))

    #     # sup = reshape_x(support)
    #     # qry = reshape_x(query)
    #     # sup_emb = self.encoder(sup,  training=training)  # (N*K, D)
    #     # qry_emb = self.encoder(qry,  training=training)  # (N*Q, D)

    #     # sup_emb = tf.reshape(sup_emb, (N, K, -1))
    #     # prototypes = tf.reduce_mean(sup_emb, axis=1)  # (N, D)

    #     def euclidean_dist(a, b):
    #         a_exp = tf.expand_dims(a, 1)
    #         b_exp = tf.expand_dims(b, 0)
    #         return tf.reduce_sum(tf.square(a_exp - b_exp), axis=2)

    #     # dists = euclidean_dist(qry_emb, prototypes)  # (N*Q, N)
    #     dists = euclidean_dist(qry_emb, sup_emb)  # (N*Q, N*K)
    #     return tf.nn.log_softmax(-dists, axis=1)

# --- Training Loop ---
def train_prototypical(model, dataset, optimizer, N, K, Q, epochs=20):
    for epoch in range(1, epochs + 1):
        total_loss, total_acc = 0.0, 0.0
        count = 0
        for support_x, support_y, query_x, query_y in dataset:
            count += 1
            with tf.GradientTape() as tape:
                log_p_y = model(support_x, query_x, training=True)
                indices = tf.stack([tf.range(N*Q), query_y], axis=1)
                loss = -tf.reduce_mean(tf.gather_nd(log_p_y, indices))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            preds = tf.argmax(log_p_y, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
            total_loss += loss.numpy()
            total_acc += acc.numpy()
        print(f"Epoch {epoch}: Loss={total_loss/count:.4f}, Acc={total_acc/count:.4f}")

# --- Main Script ---
if __name__ == '__main__':
    from glob import glob

    # # conGD
    # all_paths = glob(CONGD_OUTPUT_DIR + '/*.npy')
    # mapping = {}
    # for p in all_paths:
    #     label = os.path.basename(p).split('_', 5)[-1][3:].rsplit('.npy', 1)[0]  # "lblXXX"
    #     mapping.setdefault(label, []).append(p)

    # jester
    base_dir = JESTER_OUTPUT_DIR
    train_paths = glob(os.path.join(base_dir, 'Train', '*.npy'))
    test_paths  = glob(os.path.join(base_dir, 'Validation', '*.npy'))
    all_paths = train_paths + test_paths
    print(f"Found {len(all_paths)} total gesture paths.")

    # build label->paths
    mapping = {}
    for p in all_paths:
        label = os.path.basename(p).split('_', 1)[1].rsplit('.npy',1)[0]
        if label == 'None': continue
        mapping.setdefault(label, []).append(p)

    # filter classes and ensure enough for episode
    N, K, Q = 10, 2, 2
    mapping = {cls: ps for cls, ps in mapping.items() if len(ps) >= (K+Q)}   

    # dataset
    episodes = 2000
    epi_ds = GestureEpisodeDataset(mapping, N, K, Q, episodes).get_tf_dataset()

    # model & optimizer
    model = PrototypicalNetwork(N=N, K=K, Q=Q, embedding_dim=64)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # train
    train_prototypical(model, epi_ds, optimizer, N, K, Q, epochs=20)

