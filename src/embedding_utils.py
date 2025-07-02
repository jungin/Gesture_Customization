import os
import numpy as np
import tensorflow as tf

def extract_embeddings_stgcn(model, dataset):
    embeddings, labels = [], []
    for _, _, qry_x, qry_y in dataset:
        emb = model.encoder(qry_x, training=False)
        embeddings.append(emb.numpy())
        labels.append(qry_y.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

def extract_embeddings_cnn(model, dataset):
    embeddings, labels = [], []
    for _, _, qry_x, qry_y in dataset:
        B, T, V, C = qry_x.shape
        qry = tf.reshape(qry_x, [B, V, C * T])
        qry = tf.expand_dims(qry, -1)
        emb = model.encoder(qry, training=False)
        embeddings.append(emb.numpy())
        labels.append(qry_y.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

def save_embeddings_to_npy(embeddings, labels, prefix, out_dir='umap_outputs'):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{prefix}_embeddings.npy"), embeddings)
    np.save(os.path.join(out_dir, f"{prefix}_labels.npy"), labels)
