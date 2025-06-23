import os
import numpy as np
import tensorflow as tf
import random
from src.data_loader import GestureEpisodeDataset
from src.variables import *  # FRAMES, etc.

# --- Prototypical Network ---
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, N, K, Q, embedding_dim=128):
        super().__init__()
        self.N, self.K, self.Q = N, K, Q
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, support, query, training=False):
        def reshape_x(x):
            b, T, V, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            x_ = tf.transpose(x, [0,2,3,1])
            return tf.reshape(x_, (b, V, C * T, 1))

        sup = reshape_x(support)
        qry = reshape_x(query)
        sup_emb = self.encoder(sup,  training=training)  # (N*K, D)
        qry_emb = self.encoder(qry,  training=training)  # (N*Q, D)

        sup_emb = tf.reshape(sup_emb, (N, K, -1))
        prototypes = tf.reduce_mean(sup_emb, axis=1)  # (N, D)

        def euclidean_dist(a, b):
            a_exp = tf.expand_dims(a, 1)
            b_exp = tf.expand_dims(b, 0)
            return tf.reduce_sum(tf.square(a_exp - b_exp), axis=2)

        dists = euclidean_dist(qry_emb, prototypes)  # (N*Q, N)
        return tf.nn.log_softmax(-dists, axis=1)

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

    # conGD
    all_paths = glob(CONGD_OUTPUT_DIR + '/*.npy')
    mapping = {}
    for p in all_paths:
        label = os.path.basename(p).split('_', 5)[-1][3:].rsplit('.npy', 1)[0]  # "lblXXX"
        mapping.setdefault(label, []).append(p)

    # # jester
    # base_dir = JESTER_OUTPUT_DIR
    # train_paths = glob(os.path.join(base_dir, 'Train', '*.npy'))
    # test_paths  = glob(os.path.join(base_dir, 'Validation', '*.npy'))
    # all_paths = train_paths + test_paths
    # print(f"Found {len(all_paths)} total gesture paths.")

    # # build label->paths
    # mapping = {}
    # for p in all_paths:
    #     label = os.path.basename(p).split('_', 1)[1].rsplit('.npy',1)[0]
    #     if label == 'None': continue
    #     mapping.setdefault(label, []).append(p)

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

