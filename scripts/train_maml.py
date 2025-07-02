import os
from glob import glob
import random
import tensorflow as tf
import numpy as np

from src.data_loader import GestureEpisodeDataset
from src.variables import *  # FRAMES, etc.
from src.st_gcn import STGCN  # 위에서 만든 stgcn_tf.py
from src.utility import build_two_hand_adjacency

# 1) 사전훈련된 ST-GCN 로드
A = build_two_hand_adjacency()
base_model = STGCN(in_channels=3, num_class=27, A=A, num_layers=9)
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
_ = base_model(tf.zeros((1, 3, 42, 37)))  # build
base_model.load_weights('models/st_gcn_jester_best_model.keras')

# 2) feature_extractor 생성 & freeze
feature_extractor = tf.keras.Sequential(base_model.blocks + [base_model.global_pool])
feature_extractor.trainable = False
_ = feature_extractor(tf.zeros((1, 3, 42, 37)))

class MAMLHead(tf.Module):
    def __init__(self, feature_extractor, emb_dim, way):
        super().__init__()
        self.fe = feature_extractor
        # inner/outer-loop에서만 업데이트할 head 파라미터
        self.w = tf.Variable(tf.random.normal([emb_dim, way]), name='w')
        self.b = tf.Variable(tf.zeros([way]), name='b')

    def __call__(self, x, params=None):
        # x: (batch, T, V, C) → (batch, C, V, T)
        x = tf.transpose(x, [0,3,2,1])
        feat = self.fe(x, training=False)           # (batch, emb_dim)
        w, b = params if params is not None else (self.w, self.b)
        return tf.matmul(feat, w) + b               # logits


# loss function
def sparse_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def maml_step(model, opt, sx, sy, qx, qy, inner_steps=1, alpha=0.01):
    # outer‐tape: 메타 업데이트
    with tf.GradientTape() as outer_tape:
        fw, fb = model.w, model.b

        # inner‐loop: fast weights 계산
        for _ in range(inner_steps):
            with tf.GradientTape() as inner_tape:
                # fw, fb가 Tensor여도 감시하도록
                inner_tape.watch([fw, fb])
                logits_s = model(sx, params=[fw, fb])
                ls = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy, logits=logits_s)
                )
            grads = inner_tape.gradient(ls, [fw, fb])
            # 여기선 grads가 None이 아님
            fw = fw - alpha * grads[0]
            fb = fb - alpha * grads[1]

        # query 손실
        logits_q = model(qx, params=[fw, fb])
        lq = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=qy, logits=logits_q)
        )

    # outer‐gradient
    meta_grads = outer_tape.gradient(lq, [model.w, model.b])
    opt.apply_gradients(zip(meta_grads, [model.w, model.b]))
    return ls, lq



# Usage
def meta_train(meta_ds, emb_dim, way, epochs=100, inner=1, alpha=0.01):
    maml = MAMLHead(feature_extractor, emb_dim, way)
    opt  = tf.keras.optimizers.Adam(1e-3)
    for e in range(1, epochs+1):
        tot_lq = 0.0
        tot_acc = 0.0
        cnt = 0

        for sx, sy, qx, qy in meta_ds:
            # inner+outer update
            ls, lq = maml_step(maml, opt, sx, sy, qx, qy, inner, alpha)
            tot_lq += lq.numpy()

            # inner-loop 적응 후 query accuracy 계산
            fw, fb = maml.w, maml.b
            for _ in range(inner):
                with tf.GradientTape() as tape2:
                    tape2.watch([fw, fb])
                    logits_s = maml(sx, params=[fw, fb])
                    loss_s   = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=sy, logits=logits_s))
                grads = tape2.gradient(loss_s, [fw, fb])
                fw, fb = fw - alpha * grads[0], fb - alpha * grads[1]

            logits_q = maml(qx, params=[fw, fb])
            preds_q  = tf.argmax(logits_q, axis=1, output_type=tf.int32)
            acc_q    = tf.reduce_mean(tf.cast(preds_q == qy, tf.float32)).numpy()

            tot_acc += acc_q
            cnt += 1

        print(f"[Epoch {e}] Meta-loss: {tot_lq/cnt:.4f}, Meta-acc: {tot_acc/cnt:.4f}")

    return maml


def eval_maml(model, test_ds, inner=1, alpha=0.01):
    total_acc = 0.0
    count = 0

    for sx, sy, qx, qy in test_ds:
        # Inner-loop 적응
        fw, fb = model.w, model.b
        for _ in range(inner):
            with tf.GradientTape() as tape2:
                logits_s = model(sx, params=[fw, fb])
                loss_s   = sparse_loss(logits_s, sy)
            grads = tape2.gradient(loss_s, [fw, fb])
            fw, fb = fw - alpha * grads[0], fb - alpha * grads[1]

        # 적응된 파라미터로 query 평가
        logits_q = model(qx, params=[fw, fb])
        preds_q  = tf.argmax(logits_q, axis=1, output_type=tf.int32)
        acc_q    = tf.reduce_mean(tf.cast(preds_q == qy, tf.float32)).numpy()

        total_acc += acc_q
        count += 1

    print(f"Meta-test Few-Shot Accuracy: {total_acc/count:.4f}")


if __name__ == "__main__":
    # ————————————————————————————————————————
    # 1) 데이터 준비
    # ————————————————————————————————————————
    # conGD 파일 로드
    all_paths = glob(CONGD_OUTPUT_DIR + '/*.npy')
    mapping = {}
    for p in all_paths:
        label = os.path.basename(p).split('_', 5)[-1][3:].rsplit('.npy', 1)[0]
        mapping.setdefault(label, []).append(p)

    # 클래스 분할
    classes = sorted(mapping.keys())
    random.seed(42)
    train_classes = random.sample(classes, 20)
    test_classes  = [c for c in classes if c not in train_classes]

    train_mapping = {c: mapping[c] for c in train_classes}
    test_mapping  = {c: mapping[c] for c in test_classes}

    # meta-episode dataset 생성
    N, K, Q = 10, 2, 2
    train_ds = GestureEpisodeDataset(train_mapping, N, K, Q, 2000).get_tf_dataset()
    test_ds  = GestureEpisodeDataset(test_mapping,  N, K, Q, 500).get_tf_dataset()

    # ————————————————————————————————————————
    # 2) Meta-training & Meta-test 호출
    # ————————————————————————————————————————
    emb_dim = feature_extractor.output_shape[-1]  # 예: 64
    maml_head = meta_train(
        meta_ds=train_ds,
        emb_dim=emb_dim,
        way=N,
        epochs=200,
        inner=5,
        alpha=0.01
    )

    eval_maml(maml_head, test_ds, inner=5, alpha=0.01)