import tensorflow as tf
from .st_gcn import STGCN
from src.utility import build_two_hand_adjacency

class STGCNEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=128, num_layers=4):
        super().__init__()
        A = build_two_hand_adjacency()
        print(f"Adjacency matrix shape: {A.shape}")  # (42, 42)
        self.model = STGCN(
            in_channels=3,
            num_class=embedding_dim,  # 여기서는 classification이 아니라 embedding 용도
            A=A,
            num_layers=num_layers
        )

    def call(self, x, training=False):
        # x: (B, T, V, C) → transpose to (B, C, V, T)
        x = tf.transpose(x, [0, 3, 2, 1])
        return self.model(x, training=training)
    
class ProtoNetSTGCN(tf.keras.Model):
    def __init__(self, N, K, Q, embedding_dim=128, num_layers=4):
        super().__init__()
        self.N, self.K, self.Q = N, K, Q
        self.encoder = STGCNEncoder(embedding_dim, num_layers)

    def call(self, support, query, training=False):
        sup_emb = self.encoder(support, training=training)
        qry_emb = self.encoder(query, training=training)

        sup_emb = tf.reshape(sup_emb, (self.N, self.K, -1))
        prototypes = tf.reduce_mean(sup_emb, axis=1)
        
        # Euclidean distance
        a = tf.expand_dims(qry_emb, 1)
        b = tf.expand_dims(prototypes, 0)
        dists = tf.reduce_sum((a - b) ** 2, axis=2)
        return tf.nn.log_softmax(-dists, axis=1)

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

