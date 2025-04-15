import os
import math
import pandas as pd
import numpy as np
import tensorflow as tf

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, csv, dir, batch_size: int = 32, shuffle=False, num_frames=None, num_joints=None, coordinate_dim=None):
        self.df = pd.read_csv(csv)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dir = dir
        
        # 더미 데이터 생성에 필요한 속성 추가
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.coordinate_dim = coordinate_dim

        # Unique classes 설정
        classes = sorted(self.df.label_id.unique())
        self.class_map = {label: idx for idx, label in enumerate(classes)}
        self.num_classes = len(classes)

        # 파일 경로와 라벨 준비
        self.files = self.__get_files(dir, self.df)
        self.labels = self.__get_labels(self.df)
        print("label shape:", self.labels.shape)
        print("file shape:", self.files.shape) 

        # 데이터 셔플
        self.on_epoch_end()

    def __get_files(self, dir, df):
        files = []
        for filename in df.video_id:
            filepath = os.path.join(dir, str(filename) + '.npy')
            files.append(filepath)     
        return np.asarray(files)
   
    def __get_labels(self, df):
        labels = []
        for i in df.label_id:
            # 라벨을 원-핫 인코딩하여 배열에 추가
            label = np.zeros(self.num_classes)
            col = self.class_map[i]  # 클래스 맵을 이용하여 인덱스 찾기
            label[col] = 1.0
            labels.append(label)
        return np.asarray(labels)

    def on_epoch_end(self):
        # 에포크가 끝날 때마다 데이터 셔플
        if self.shuffle:
            indices = np.arange(len(self.files))
            np.random.shuffle(indices)
            self.files = self.files[indices]
            self.labels = self.labels[indices]
    
    def __getitem__(self, index):
        # 주어진 인덱스에 해당하는 배치 데이터 반환
        batches = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__get_data(batches)
        return X, y
    
    def __get_data(self, batches):
        X_batch = []
        expected_shape = None
        
        for file in batches:
            if os.path.exists(file):
                try:
                    video_arr = np.load(file)
                    
                    # 첫 번째 유효한 배열의 shape을 기준으로 설정
                    if expected_shape is None:
                        expected_shape = video_arr.shape
                    
                    # shape이 일치하지 않는 경우 리사이징 또는 패딩
                    if video_arr.shape != expected_shape:
                        print(f"Warning: File {file} has shape {video_arr.shape}, expected {expected_shape}")
                        # 여기서 리사이징이나 패딩 적용 (예시는 생략)
                        # 또는 expected_shape에 맞는 더미 데이터 사용
                        if all(dim is not None for dim in [self.num_frames, self.num_joints, self.coordinate_dim]):
                            video_arr = np.zeros(expected_shape)
                        else:
                            video_arr = np.zeros(expected_shape)
                    
                    X_batch.append(video_arr)
                except Exception as e:
                    print(f"Error loading file {file}: {e}")
                    if expected_shape is not None:
                        X_batch.append(np.zeros(expected_shape))
                    elif all(dim is not None for dim in [self.num_frames, self.num_joints, self.coordinate_dim]):
                        expected_shape = (self.num_frames, self.num_joints, self.coordinate_dim)
                        X_batch.append(np.zeros(expected_shape))
                    else:
                        raise ValueError(f"Cannot create dummy data for {file}: dimensions not specified and no valid samples found")
            else:
                print(f"Warning: File {file} does not exist.")
                if expected_shape is not None:
                    X_batch.append(np.zeros(expected_shape))
                elif all(dim is not None for dim in [self.num_frames, self.num_joints, self.coordinate_dim]):
                    expected_shape = (self.num_frames, self.num_joints, self.coordinate_dim)
                    X_batch.append(np.zeros(expected_shape))
                else:
                    raise ValueError(f"Cannot create dummy data for {file}: dimensions not specified and no valid samples found")
        
        return np.array(X_batch)
        
    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)