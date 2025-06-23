import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import HeNormal, GlorotUniform

class GraphConvolution(layers.Layer):
    """
    Simple GCN layer with fixed adjacency matrix stored as a non-trainable weight.
    Input: (batch, C, V, T)
    Output: (batch, out_channels, V, T)
    """
    def __init__(self, out_channels, A, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.A_init = np.array(A, dtype=np.float32)
        self.use_bias = use_bias

    def build(self, input_shape):
        V = self.A_init.shape[0]
        self.A = self.add_weight(
            name='A', shape=(V, V),
            initializer=tf.keras.initializers.Constant(self.A_init),
            trainable=False
        )
        self.conv = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=(1,1),
            use_bias=self.use_bias,
            data_format='channels_first'
        )
        super().build(input_shape)

    def call(self, x):
        # (batch, C, V, T) -> (batch, C, T, V)
        x_t = tf.transpose(x, [0,1,3,2])
        x_t = tf.einsum('nctv,vw->nctw', x_t, self.A)
        x_t = self.conv(x_t)
        return tf.transpose(x_t, [0,1,3,2])  # back to (batch, C, V, T)

class STGCNBlock(layers.Layer):
    """
    One ST-GCN block: spatial GCN -> temporal conv -> residual + ReLU
    """
    def __init__(self, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.A = A
        self.stride = stride
        self.residual_flag = residual

    def build(self, input_shape):
        in_channels = input_shape[1]
        self.gcn = GraphConvolution(self.out_channels, self.A)
        # self.bn1 = layers.BatchNormalization(axis=1)
        self.ln1 = layers.LayerNormalization(axis=[1,2,3])
        self.act = layers.ReLU()
        # Temporal Conv
        self.tcn_conv = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=(9,1),
            strides=(self.stride,1),  # temporal stride on T axis after transpose
            padding='same',
            data_format='channels_first',
            kernel_initializer=HeNormal(),
            bias_initializer='zeros'
        )
        # self.bn2 = layers.BatchNormalization(axis=1)
        self.ln2 = layers.LayerNormalization(axis=[1,2,3])
        self.dropout = layers.Dropout(0.3)

        # Residual: match shape (batch, out_channels, V, T')
        if in_channels == self.out_channels and self.stride == 1:
            self.residual_layer = None
        else:
            # Use conv on (batch, C, V, T): need to stride on T axis -> strides=(1, stride)
            self.residual_layer = tf.keras.Sequential([
                # Conv2D sees height=V, width=T
                layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=(1,1),
                    strides=(1, self.stride),  # no change on V, stride on T
                    data_format='channels_first'
                ),
                layers.BatchNormalization(axis=1)
            ])
        super().build(input_shape)

    def call(self, x, training=False):
        x_in = x
        # Spatial conv
        x1 = self.gcn(x_in)
        # Temporal conv: transpose to (batch, C, T, V)
        x1 = tf.transpose(x1, [0,1,3,2])
        x1 = self.ln1(x1)
        # x1 = self.bn1(x1, training=training)
        x1 = self.act(x1)
        x1 = self.tcn_conv(x1)
        x1 = self.ln2(x1)
        x1 = self.dropout(x1, training=training)
        # x1 = self.bn2(x1, training=training)

        # Back to (batch, C, V, T)
        x1 = tf.transpose(x1, [0,1,3,2])
        # Residual path
        if not self.residual_flag:
            res = tf.zeros_like(x1)
        else:
            if self.residual_layer is None:
                res = x_in
            else:
                res = self.residual_layer(x_in, training=training)
        x_out = x1 + res
        return self.act(x_out)

class STGCN(Model):
    """
    Spatial-Temporal GCN for skeleton-based action recognition.
    """
    def __init__(self, in_channels, num_class, A, num_layers=9, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        c1 = 32
        self.blocks.append(STGCNBlock(c1, A, stride=1, residual=False))
        for i in range(1, num_layers):
            stride = 2 if i % 3 == 0 else 1
            self.blocks.append(STGCNBlock(c1, A, stride=stride, residual=True))
        self.global_pool = layers.GlobalAveragePooling2D(data_format='channels_first')
        self.fc = layers.Dense(num_class)

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        x = self.global_pool(x)
        return self.fc(x)

if __name__ == '__main__':
    B, C, V, T = 8, 3, 42, 37
    A = np.eye(V, dtype=np.float32)
    model = STGCN(in_channels=C, num_class=27, A=A, num_layers=9)
    dummy = tf.random.normal((B, C, V, T))
    logits = model(dummy, training=False)
    print('Output shape:', logits.shape)
