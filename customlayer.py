import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Dropout


# 공통으로 사용할 엣지 정보를 가진 베이스 클래스 생성
class HandGraphBase:
    @staticmethod
    def get_hand_edges():
        # 왼손 관절 연결 정보
        left_hand_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # Index Finger
            (0, 9), (9, 10), (10, 11), (11, 12),    # Middle Finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
            (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky Fingcler
        ]

        # 오른손 관절 연결 정보
        right_hand_edges = [
            (21, 22), (22, 23), (23, 24), (24, 25),  # Thumb
            (21, 26), (26, 27), (27, 28), (28, 29),  # Index Finger
            (21, 30), (30, 31), (31, 32), (32, 33),  # Middle Finger
            (21, 34), (34, 35), (35, 36), (36, 37),  # Ring Finger
            (21, 38), (38, 39), (39, 40), (40, 41)   # Pinky Finger
        ]

        # 같은 손의 손가락 끝마디 간 연결
        same_hand_finger_tips = [
            (4, 8), (8, 12), (12, 16), (16, 20),  # 왼손
            (25, 29), (29, 33), (33, 37), (37, 41)  # 오른손
        ]

        # 양손 간 연결
        cross_hand_edges = [(0, 21)]

        return left_hand_edges + right_hand_edges + cross_hand_edges + same_hand_finger_tips

    @staticmethod
    def create_adjacency_matrix():
        all_edges = HandGraphBase.get_hand_edges()
        num_nodes = 42
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        for edge in all_edges:
            adjacency_matrix[edge[0], edge[1]] = 1
            adjacency_matrix[edge[1], edge[0]] = 1
            
        return tf.convert_to_tensor(adjacency_matrix, dtype=tf.float32)


# 인접 행렬을 모델 내부에서 상수로 반환하는 Lambda 레이어
@keras.saving.register_keras_serializable(package='Custom', name='get_adjacency_matrix')
class GetAdjacencyMatrixLayer(Layer, HandGraphBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adjacency_matrix = self.create_adjacency_matrix()

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        return self.adjacency_matrix


@keras.saving.register_keras_serializable(package='Custom', name='GraphTransformerLayer')
class GraphTransformerLayer(Layer):
    def __init__(self, embed_dim, num_heads, max_distance=7, **kwargs):
        super(GraphTransformerLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_distance = max_distance
        
        # 기존 Dense 레이어들
        self.query_dense = Dense(self.embed_dim)
        self.key_dense = Dense(self.embed_dim)
        self.value_dense = Dense(self.embed_dim)
        self.attention_dropout = Dropout(0.1)
        self.feature_dropout = Dropout(0.1)
        self.output_dense = Dense(self.embed_dim)
        
        # 상대적 위치 인코딩을 위한 임베딩 레이어 추가
        self.relative_embedding = tf.keras.layers.Embedding(
            2 * max_distance + 1,  # -max_distance ~ +max_distance
            num_heads
        )
    
    def build(self, input_shape):
        # 기존 build 로직
        if not isinstance(input_shape, (list, tuple)):
            input_shape = [input_shape]
        
        node_features_shape = input_shape[0]
        if node_features_shape is None:
            return

        self.query_dense.build(tf.TensorShape([None, None, None, node_features_shape[-1]]))
        self.key_dense.build(tf.TensorShape([None, None, None, node_features_shape[-1]]))
        self.value_dense.build(tf.TensorShape([None, None, None, node_features_shape[-1]]))  
        
        output_shape = tf.TensorShape([None, None, None, self.embed_dim])
        self.output_dense.build(output_shape)
        
        # 관절 간의 거리 행렬 추가
        distances = np.zeros((42, 42))
        for edge in HandGraphBase.get_hand_edges():  # HandGraphBase의 메소드 사용
            distances[edge[0], edge[1]] = 1
            distances[edge[1], edge[0]] = 1
        
        # Floyd-Warshall 알고리즘으로 최단 경로 거리 계산
        for k in range(42):
            for i in range(42):
                for j in range(42):
                    if i != j:
                        # 중요: k를 통해 가는 경로가 있는 경우만 확인 (둘 다 0이 아님)
                        if distances[i,k] > 0 and distances[k,j] > 0:
                            new_dist = distances[i,k] + distances[k,j]
                             # 핵심: 아직 경로가 없거나(0) 더 짧은 경로를 찾은 경우
                            if distances[i,j] == 0 or new_dist < distances[i,j]:
                                distances[i,j] = new_dist
        
        # 최대 거리로 클리핑
        distances = np.clip(distances, 0, self.max_distance)
        self.distances = tf.convert_to_tensor(distances, dtype=tf.int32)
        
        self.built = True

    def call(self, node_features, adjacency_matrix):
        batch_size = tf.shape(node_features)[0]
        time_steps = tf.shape(node_features)[1]
        num_nodes = tf.shape(node_features)[2]
        
        # Q, K, V 계산: [batch_size, time_steps, num_nodes, num_heads * head_dim]
        Q = self.query_dense(node_features)
        K = self.key_dense(node_features)
        V = self.value_dense(node_features)
        
        # Multi-head 분리: [batch_size, time_steps, num_nodes, num_heads, head_dim]
        Q = tf.reshape(Q, [batch_size, time_steps, num_nodes, self.num_heads, -1])
        K = tf.reshape(K, [batch_size, time_steps, num_nodes, self.num_heads, -1])
        V = tf.reshape(V, [batch_size, time_steps, num_nodes, self.num_heads, -1])
        
        # Multi-head attention을 위해 Q, K, V를 트랜스포즈: [batch_size, time_steps, num_heads, num_nodes, head_dim]
        Q = tf.transpose(Q, [0, 1, 3, 2, 4])
        K = tf.transpose(K, [0, 1, 3, 2, 4])
        V = tf.transpose(V, [0, 1, 3, 2, 4])
        
        # attention 계산
        attention_scores = tf.matmul(Q, K, transpose_b=True) # [batch_size, time_steps, num_heads, num_nodes, num_nodes]
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # 상대적 위치 인코딩 추가
        # relative_positions = self.distances + self.max_distance           # 음수 거리를 양수로 변환
        relative_positions = tf.expand_dims(self.distances, axis=0)         # [1, nodes, nodes]
        relative_bias = self.relative_embedding(relative_positions)         # [1, nodes, nodes, heads]
        relative_bias = tf.transpose(relative_bias, [0, 3, 1, 2])           # [1, heads, nodes, nodes]
        
        # Add relative position bias to attention scores
        attention_scores = attention_scores + tf.expand_dims(relative_bias, axis=1)
        
        # 기존 마스킹 및 나머지 로직
        adjacency_mask = tf.cast(adjacency_matrix > 0, dtype=tf.float32)
        adjacency_mask = tf.expand_dims(adjacency_mask, axis=0)
        adjacency_mask = tf.expand_dims(adjacency_mask, axis=0)
        adjacency_mask = tf.expand_dims(adjacency_mask, axis=0)
        
        attention_scores = attention_scores * adjacency_mask + (1.0 - adjacency_mask) * (-1e9)
        
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        output = tf.matmul(attention_weights, V) 
        output = tf.transpose(output, [0, 1, 3, 2, 4])
        output = tf.reshape(output, [batch_size, time_steps, num_nodes, self.embed_dim])
        
        output = self.output_dense(output)
        
        return output, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "max_distance": self.max_distance, 
        })
        return config


@keras.saving.register_keras_serializable(package='Custom', name='ReshapeLayer1')
class ReshapeLayer1(Layer):
    def __init__(self, **kwargs):
        super(ReshapeLayer1, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.built = True
    
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        num_nodes = input_shape[2]
        features = input_shape[3]
        
        x = tf.transpose(inputs, [0, 2, 1, 3])  # (batch, nodes, time, features)
        x = tf.reshape(x, [batch_size, num_nodes, time_steps * features])
        
        return x
    
    def compute_output_shape(self, input_shape):
        # None 체크를 가장 먼저
        if input_shape is None:
            return tf.TensorShape([None, None, None])
        
        # 리스트나 튜플 처리
        if isinstance(input_shape, (list, tuple)):
            if not input_shape:  # 빈 리스트/튜플 체크
                return tf.TensorShape([None, None, None])
            input_shape = input_shape[0]
        
        # None인 경우 다시 체크
        if input_shape is None:
            return tf.TensorShape([None, None, None])
        
        # TensorShape 처리
        if isinstance(input_shape, tf.TensorShape):
            if input_shape.rank is None:  # rank가 None인 경우
                return tf.TensorShape([None, None, None])
                
            dims = input_shape.as_list()
            if len(dims) != 4 or any(d is None for d in dims):
                return tf.TensorShape([None, None, None])
                
            return tf.TensorShape([dims[0], dims[2], dims[1] * dims[3]])
        
        # 일반 리스트/튜플 shape 처리
        try:
            if len(input_shape) != 4:
                return tf.TensorShape([None, None, None])
            return tf.TensorShape([input_shape[0], input_shape[2], input_shape[1] * input_shape[3]])
        except (TypeError, AttributeError):
            return tf.TensorShape([None, None, None])
    
    def get_config(self):
        return super().get_config()
    

@keras.saving.register_keras_serializable(package='Custom', name='ReshapeLayer2')
class ReshapeLayer2(Layer):
    """3D 텐서를 2D로 변환하는 커스텀 reshape 레이어.
    
    입력 텐서 (batch, nodes, time*features)를 받아서
    고정된 차원의 2D 텐서로 변환합니다.
    """
    
    def __init__(self, **kwargs):
        super(ReshapeLayer2, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if input_shape is None or len(input_shape) != 3:
            raise ValueError("Expected 3D input shape (batch, nodes, features)")
        self.built = True
    
    def call(self, inputs):
        # 입력을 (batch, nodes, features)로 해석
        batch_size = tf.shape(inputs)[0]
        
        # 고정된 차원으로 변환 (42 nodes * 2368 features = 99456)
        x = tf.reshape(inputs, [batch_size, 42 * 2368])
        return x
    
    def compute_output_shape(self, input_shape):
        # 출력 shape을 명시적으로 지정
        return tf.TensorShape([input_shape[0], 42 * 2368])
    
    def get_config(self):
        return super().get_config()