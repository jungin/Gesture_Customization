
import os
from glob import glob
import numpy as np
import mediapipe as mp
import tensorflow as tf
from src.variables import *


def build_metadata_jester(root_dir, pattern="*.npy"):
    """
    root_dir 아래의 모든 .npy 파일을 찾고,
    파일명이나 디렉토리 구조에서 라벨을 추출하여
    [(path1, label1), (path2, label2), ...] 형태의 리스트를 반환.
    """
    meta = []
    inv = {v: k for k, v in JESTER_LABELS.items()}
    for filepath in glob(os.path.join(root_dir, "**", pattern), recursive=True):
        fname = os.path.basename(filepath)
        name, _ = os.path.splitext(fname)           # "00001_SwipeLeft"
        label = name.split("_", 1)[1]               # "SwipeLeft"
        label = inv[label]  
        meta.append((filepath, label))
    return meta

def parse_fn(np_path, label):
    path = np_path.decode('utf-8')
    seq = np.load(path)                 # (T, V, 3)

    # 1) 표준화
    mean = seq.mean(axis=(0,1), keepdims=True)
    std  = seq.std(axis=(0,1), keepdims=True)
    seq = (seq - mean) / (std + 1e-6)   # (T, V, 3)

    # 2) 롤·턴 방향성 피처 추가 (총 7채널)
    seq = augment_features(seq)         # (T, V, 7)

    # 3) (T, V, C) → (C, V, T)
    seq = np.transpose(seq, (2,1,0)).astype(np.float32)  # (7, V, T)
    return seq, label

def tf_parse_fn(np_path, label):
    seq, lbl = tf.numpy_function(parse_fn, [np_path, label],
                                 [tf.float32, tf.int32])
    # 채널 수를 7로 변경
    seq.set_shape((7, 42, 37))
    lbl.set_shape(())
    return seq, lbl


# MediaPipe Hands 초기화 (양손)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    model_complexity=1
)

def pad_or_trim(seq, target_len):
    """
    seq: ndarray of shape (L, 42, 3)
    target_len: 원하는 길이 L_max
    returns: ndarray of shape (target_len, 42, 3)
    """
    L, D1, D2 = seq.shape
    if L < target_len:
        # 앞뒤 0 패딩: 앞 0·뒤 (target_len - L) 0
        pad_front = np.zeros((target_len - L, D1, D2), dtype=seq.dtype)
        return np.concatenate([seq, pad_front], axis=0)
    else:
        # 균등 간격 샘플링 (예: 첫·마지막 포함)
        idxs = np.linspace(0, L-1, num=target_len).astype(int)
        return seq[idxs]
    

def compute_joint_movement_variance(seq):
    diffs = np.diff(seq, axis=0)  # (T-1, 42, 3)
    movement = np.linalg.norm(diffs, axis=2)  # (T-1, 42)
    return movement.var()


def interpolate_missing_frames(seq):
    """
    seq: ndarray of shape [T, 42, 3]
    Returns:
      new_seq: same shape, but 모든-0 프레임을 선형 보간으로 채움
    """
    if seq.size == 0:
        return seq
    T, J, C = seq.shape
    frame_idx = np.arange(T)
    new_seq = seq.copy()

    # 각 joint, 각 coord 별로 보간
    for j in range(J):
        for c in range(C):
            ts = new_seq[:, j, c]
            # valid 프레임(0이 아닌) 인덱스
            valid = np.where(ts != 0)[0]
            # valid 없으면 통과
            if valid.size == 0:
                continue
            # 맨 앞/뒤가 missing이면 가장 가까운 valid 값으로 채워주기
            if valid[0] > 0:
                ts[:valid[0]] = ts[valid[0]]
            if valid[-1] < T-1:
                ts[valid[-1]+1:] = ts[valid[-1]]
            # 실제 보간
            new_seq[:, j, c] = np.interp(frame_idx, valid, ts[valid])
    return new_seq


def compute_palm_normals(seq):
    """
    Compute palm normal vectors for each frame based on wrist, index MCP (5), pinky MCP (17).
    seq: (T, 42, 3)
    Returns: normals: (T, 3)
    """
    wrist = seq[:, 0, :]       # (T,3)
    i_mcp = seq[:, 5, :]       # (T,3)
    p_mcp = seq[:, 17, :]      # (T,3)
    v1 = i_mcp - wrist
    v2 = p_mcp - wrist
    normals = np.cross(v1, v2)  # (T,3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-6)
    return normals

def compute_roll_angles(normals):
    """
    Compute signed roll angles relative to first frame normal.
    normals: (T,3)
    Returns: angles: (T,)
    """
    ref = normals[0]
    # cos(theta) = dot(ref, normals)
    cos = np.clip(np.dot(normals, ref), -1.0, 1.0)
    angles = np.arccos(cos)  # [0, π]
    # determine sign via cross product direction
    cross = np.cross(ref, normals)  # (T,3)
    sign = np.sign(np.dot(cross, ref))  # positive if same direction
    return angles * sign  # (-π, π)

def compute_rotation_angles(seq):
    """
    Compute hand rotation angles based on wrist -> index tip vector.
    seq: (T, 42, 3)
    Returns: angles: (T,)
    """
    # Using right hand as example: wrist index 21, index tip 29
    wrist = seq[:, 21, :2]
    idx_tip = seq[:, 29, :2]
    vec = idx_tip - wrist  # (T,2)
    angles = np.arctan2(vec[:,1], vec[:,0])  # (-π, π)
    return angles

def compute_angular_velocity(angles):
    """
    Compute wrapped angular velocity Δθ between frames.
    angles: (T,)
    Returns: vel: (T,)
    """
    diff = np.diff(angles, prepend=angles[0])
    diff = (diff + np.pi) % (2*np.pi) - np.pi
    return diff

def classify_roll(seq):
    """
    Classify Rolling gesture: 7 (Backward) or 8 (Forward).
    Returns the class index and direction.
    """
    normals = compute_palm_normals(seq)
    roll_angles = compute_roll_angles(normals)
    roll_vel = compute_angular_velocity(roll_angles)
    mean_vel = np.mean(roll_vel)
    if mean_vel > 0:
        return 8, "Rolling Forward"
    else:
        return 7, "Rolling Backward"

def classify_turn(seq):
    """
    Classify Turning gesture: 21 (Clockwise) or 22 (Counterclockwise).
    Returns the class index and direction.
    """
    angles = compute_rotation_angles(seq)
    vel = compute_angular_velocity(angles)
    mean_vel = np.mean(vel)
    if mean_vel > 0:
        return 22, "Turning Counterclockwise"
    else:
        return 21, "Turning Clockwise"

# Example usage:
# seq = np.load("path_to_seg_seq.npy")  # (37, 42, 3)
# roll_class, roll_dir = classify_roll(seq)
# turn_class, turn_dir = classify_turn(seq)
# print(f"Roll Prediction: {roll_class} ({roll_dir})")
# print(f"Turn Prediction: {turn_class} ({turn_dir})")

def augment_features(seq):
    """
    seq: (T, V, 3)
    returns augmented seq: (T, V, 5) with added roll_angle and roll_vel
    and turn_angle and turn_vel channels.
    """
    T, V, _ = seq.shape
    # Compute roll features
    normals = compute_palm_normals(seq)
    roll_angles = compute_roll_angles(normals)
    roll_vel = compute_angular_velocity(roll_angles)
    # Compute turn features
    turn_angles = compute_rotation_angles(seq)
    turn_vel = compute_angular_velocity(turn_angles)
    # Expand dims to (T, 1) then tile to joints
    roll_angle_feat = np.tile(roll_angles[:, None, None], (1, V, 1))
    roll_vel_feat   = np.tile(roll_vel[:, None, None],   (1, V, 1))
    turn_angle_feat = np.tile(turn_angles[:, None, None], (1, V, 1))
    turn_vel_feat   = np.tile(turn_vel[:, None, None],   (1, V, 1))
    # Concatenate: original (T,V,3) + roll ang & vel + turn ang & vel = (T,V,7)
    return np.concatenate([seq, roll_angle_feat, roll_vel_feat, turn_angle_feat, turn_vel_feat], axis=2)


