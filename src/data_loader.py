import os
import numpy as np
import tensorflow as tf
import random
from .variables import *

class GestureEpisodeDataset:
    def __init__(self, class_to_paths, N, K, Q, episodes_per_epoch):
        # Ensure enough classes
        if len(class_to_paths) < N:
            raise ValueError(f"Not enough classes: have {len(class_to_paths)}, need {N}")
        self.class_to_paths = class_to_paths
        self.classes = list(class_to_paths.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.episodes_per_epoch = episodes_per_epoch

    def _generator(self):
        for _ in range(self.episodes_per_epoch):
            # sample valid classes ensuring K+Q samples each
            valid = [c for c, paths in self.class_to_paths.items() if len(paths) >= self.K + self.Q]
            if len(valid) < self.N:
                raise ValueError(f"Not enough valid classes for sampling: have {len(valid)}, need {self.N}")
            selected = random.sample(valid, self.N)
            supp_x, supp_y, qry_x, qry_y = [], [], [], []
            for cls_idx, cls in enumerate(selected):
                paths = self.class_to_paths[cls]
                sampled = random.sample(paths, self.K + self.Q)
                # support
                for p in sampled[:self.K]:
                    supp_x.append(np.load(p))
                    supp_y.append(cls_idx)
                # query
                for p in sampled[self.K:]:
                    qry_x.append(np.load(p))
                    qry_y.append(cls_idx)

            yield (
                np.stack(supp_x), np.array(supp_y, dtype=np.int32),
                np.stack(qry_x), np.array(qry_y, dtype=np.int32)
            )

    def get_tf_dataset(self):
        spec = (
            tf.TensorSpec((self.N * self.K, None, 42, 3), tf.float32),
            tf.TensorSpec((self.N * self.K,), tf.int32),
            tf.TensorSpec((self.N * self.Q, None, 42, 3), tf.float32),
            tf.TensorSpec((self.N * self.Q,), tf.int32)
        )
        ds = tf.data.Dataset.from_generator(self._generator, output_signature=spec)
        return ds.prefetch(tf.data.AUTOTUNE)

# --- Helper to build mapping for a directory ---
def make_mapping_jester(dir_path):
    mapping = {}
    for fname in os.listdir(dir_path):
        if not fname.endswith('.npy'): 
            continue
        # 1) 확장자 제거 + 2) 첫 번째 언더스코어(_) 앞부분 잘라내기
        filename = os.path.splitext(fname)[0]             # "100117_No gesture"
        label = filename.split('_', 1)[1]                  # "No gesture"
        if label == 'None':
            continue
        mapping.setdefault(label, []).append(os.path.join(dir_path, fname))
    return mapping

def make_mapping_congd(dir_path):
    mapping = {}
    for fname in os.listdir(dir_path):
        if not fname.endswith('.npy'):
            continue
        name = os.path.splitext(fname)[0]             # "train(test)_folername_filename_framestart_end_lblXXX"
        label = name.split('_', 5)[-1][3:]            # "lblXXX"
        mapping.setdefault(label, []).append(os.path.join(dir_path, fname))
        print(f"{fname} -> {label}")
    return mapping

if __name__ == '__main__':
    # # congd
    # mapping = make_mapping_congd(CONGD_OUTPUT_DIR)

    # jester
    base_dir = JESTER_OUTPUT_DIR
    train_mapping = make_mapping_jester(os.path.join(base_dir, 'Train'))
    test_mapping  = make_mapping_jester(os.path.join(base_dir, 'Test'))

    # Merge train and test mappings
    mapping = {**train_mapping, **test_mapping}

    # Create one episode dataset spanning all classes
    N, K, Q = 5, 2, 2
    episodes = 100
   
    # mapping 완성 후
    min_samples = K + Q
    filtered_mapping = {
        cls: paths
        for cls, paths in mapping.items()
        if len(paths) >= min_samples
    }

    episode_ds = GestureEpisodeDataset(
        class_to_paths=filtered_mapping,
        N=N, K=K, Q=Q,
        episodes_per_epoch=episodes
    ).get_tf_dataset()

    # Example iteration
    for supp_x, supp_y, qry_x, qry_y in episode_ds.take(1):
        print('Episode:', supp_x.shape, supp_y.shape, qry_x.shape, qry_y.shape)