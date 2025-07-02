import os
import random
import json
import numpy as np
import tensorflow as tf

from src.data_loader import GestureEpisodeDataset
from src.models.protonet import ProtoNetSTGCN, ProtoNetCNN  # assume modularized

from src.variables import JESTER_OUTPUT_DIR, JESTER_LABELS
from src.utility import prepare_mapping_jester
from src.embedding_utils import extract_embeddings_stgcn, extract_embeddings_cnn, save_embeddings_to_npy

# --- Constants ---
N_FOREGROUND = 5  # number of gesture classes (excluding background)
K, Q = 1, 1
EMB_DIM = 64
TRAIN_EPISODES, TEST_EPISODES = 2000, 500
BACKGROUND_CLASS_NAMES = {'Doing other things', 'No gesture'}
class_splits_file = 'class_splits_2.json'

# --- Prepare Data Mappings ---
def prepare_split_mapping():
    mapping_all = prepare_mapping_jester(JESTER_OUTPUT_DIR, min_samples=K + Q)
    
    if not os.path.exists(class_splits_file):
        classes = list(mapping_all.keys())
        random.seed(42)
        random.shuffle(classes)
        n = len(classes)
        train_cls = classes[: int(0.6 * n)]
        val_cls = classes[int(0.6 * n): int(0.8 * n)]
        test_cls = classes[int(0.8 * n):]

        with open(class_splits_file, 'w') as f:
            json.dump({'train': train_cls, 'val': val_cls, 'test': test_cls}, f, indent=2)
    else:
        with open(class_splits_file, 'r') as f:
            splits = json.load(f)
            train_cls, val_cls, test_cls = splits['train'], splits['val'], splits['test']

    def build_mapping(cls_list):
        return {c: mapping_all[c] for c in cls_list}
        
    return (
        build_mapping(train_cls),
        build_mapping(val_cls),
        build_mapping(test_cls)
    )

# --- Training Loop ---
def train(model, train_ds, val_ds, optimizer, epochs=20):
    for epoch in range(1, epochs + 1):
        total_loss = total_acc = 0.0
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
            total_acc += acc.numpy()

        val_loss = val_acc = 0.0
        for support_x, support_y, query_x, query_y in val_ds:
            log_p = model(support_x, query_x, training=False)
            idx = tf.stack([tf.range(model.N * model.Q), query_y], axis=1)
            loss = -tf.reduce_mean(tf.gather_nd(log_p, idx))

            preds = tf.argmax(log_p, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(preds == query_y, tf.float32))
            val_loss += loss.numpy()
            val_acc += acc.numpy()

        print(f"Epoch {epoch:02d}: Train Loss={total_loss/TRAIN_EPISODES:.4f}, Train Acc={total_acc/TRAIN_EPISODES:.4f} | Val Loss={val_loss/TEST_EPISODES:.4f}, Val Acc={val_acc/TEST_EPISODES:.4f}")

# --- Evaluation ---
def evaluate(model, dataset):
    total_acc, steps = 0.0, 0
    for supp_x, supp_y, qry_x, qry_y in dataset:
        log_p = model(supp_x, qry_x, training=False)
        preds = tf.argmax(log_p, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(preds == qry_y, tf.float32))
        total_acc += acc.numpy()
        steps += 1
    print(f"Meta-test Accuracy: {total_acc/steps:.4f}")

# --- Main ---
def main():
    mapping_train, mapping_val, mapping_test = prepare_split_mapping()

    train_ds = GestureEpisodeDataset(mapping_train, N_FOREGROUND + 1, K, Q, TRAIN_EPISODES).get_tf_dataset()
    val_ds = GestureEpisodeDataset(mapping_val, N_FOREGROUND + 1, K, Q, TEST_EPISODES).get_tf_dataset()
    test_ds = GestureEpisodeDataset(mapping_test, N_FOREGROUND + 1, K, Q, TEST_EPISODES).get_tf_dataset()
    
    model = ProtoNetSTGCN(N_FOREGROUND + 1, K, Q, embedding_dim=EMB_DIM)    # ST-GCN version
    # model = ProtoNetCNN(N_FOREGROUND + 1, K, Q, embedding_dim=EMB_DIM)  # CNN version
    optimizer = tf.keras.optimizers.Adam(1e-3)

    train(model, train_ds, val_ds, optimizer, epochs=20)
    evaluate(model, test_ds)

    # for ST-GCN version
    embeds, labels = extract_embeddings_stgcn(model, test_ds)   
    save_embeddings_to_npy(embeds, labels, prefix='stgcn', out_dir='umap_outputs')

    # for CNN version
    # embeds, labels = extract_embeddings_cnn(model_cnn, test_ds)   
    # save_embeddings_to_npy(embeds, labels, prefix='cnn', out_dir='umap_outputs')

if __name__ == '__main__':
    main()
