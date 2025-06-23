
import numpy as np
import mediapipe as mp

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

