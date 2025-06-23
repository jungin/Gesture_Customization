import os
import glob
import numpy as np
import cv2
from collections import defaultdict

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

from src.variables import *
from src.utility import pad_or_trim, compute_joint_movement_variance, interpolate_missing_frames

# HandLandmarker 초기화 (global)
base_options = python.BaseOptions(
    model_asset_path=MP_HANDS_MODEL,
    delegate=python.Delegate.GPU
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     num_hands=2
# )
hand_landmarker = vision.HandLandmarker.create_from_options(options)

import threading
from concurrent.futures import ThreadPoolExecutor
thread_local = threading.local()
def get_landmarker():
    if not hasattr(thread_local, "lm"):
        options.running_mode = vision.RunningMode.VIDEO
        thread_local.lm = vision.HandLandmarker.create_from_options(options)
    return thread_local.lm

# parse segments from text file
def load_segments(txt_path, split):
    """
    Args:
      txt_path: '.../phase_1/train.txt' 등
      split:    'train' 또는 'valid' 또는 'test'
    Returns:
      dict where key is f"{split}/{video_id}"
    """
    segs = defaultdict(list)
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            video_id = parts[0]          # e.g. '048/02389'
            key = f"{split}/{video_id}"  # → 'train/048/02389' 혹은 'test/048/02389'
            for seg in parts[1:]:
                span, label = seg.split(':')
                start, end = map(int, span.split(','))
                segs[key].append((start, end, int(label)))
    return segs

# 한 비디오(.avi)에서 (T,42,3) 전체 시퀀스 뽑기
def extract_bimanual_sequence(video_path):
    """
    Args:
        video_path: .avi 파일 경로
    Returns:
        ndarray of shape (T, 42, 3), dtype float32
        — 각 프레임마다 [L0...L20, R0...R20] 순
    """
    cap = cv2.VideoCapture(video_path)
    seq = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)           # 예: 30.0
    ms_per_frame = 1000.0 / fps              # 1프레임 당 밀리초

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = mp.Image(
            mp.ImageFormat.SRGB,  # ✅ mediapipe 최상위 enum 사용
            rgb                   # contiguous uint8 ndarray
        )
        result = hand_landmarker.detect(mp_image)
        mp_image = vision.Image.create_from_array(rgb, vision.ImageFormat.SRGB)
        result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        left = [(0., 0., 0.)] * 21
        right = [(0., 0., 0.)] * 21
        
        if result.hand_landmarks and result.handedness:
            for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                score = handed[0].score
                if score < 0.5:  # or 0.4, 실험적으로 조정
                    continue  # skip low-confidence hands
                coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
                label = handed[0].category_name
                if label == "Left":
                    left = coords
                elif label == "Right":
                    right = coords

        seq.append(left + right)

    cap.release()
    return np.array(seq, dtype=np.float32)  # (T, 42, 3)

# 4) 전체 → 세그먼트로 자르고 저장
def process_congd():
    """
    폴더 구조:
    raw_data/
      ├─ ConGD_phase_1/
      │    └─ train/
      │         ├─ folder1/
      │         │    ├─ 00101.K.avi
      │         │    ├─ 00101.M.avi
      │         │    └─ …
      │         └─ folder2/…
      └─ ConGD_phase_2/
           └─ test/
                ├─ folder1/
                │    ├─ 00101.K.avi
                │    ├─ 00101.M.avi
                │    └─ …
                └─ folder2/…

    out_dir:
    skeletons/
      ├ train_001_00101_11.npy, …
      └ test_001_00101_11.npy, …
    """
    phase_1 = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_1")
    phase_2 = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_2")
    train_txt    = os.path.join(phase_1, "train.txt")
    valid_txt    = os.path.join(phase_2, "valid.txt")
    phase_1_segs = load_segments(train_txt, split="train")
    phase_2_segs  = load_segments(valid_txt, split="test")

    all_segments = {**phase_1_segs, **phase_2_segs}
    for video_id, seg_list in all_segments.items():
        print(f"Processing {video_id}...")
        split, class_folder, vid_name = video_id.split('/')
        if split == 'train':
            avi_path = os.path.join(CONGD_INPUT_DIR, 'ConGD_phase_1', split, class_folder, vid_name + '.M.avi')
        else:
            avi_path = os.path.join(CONGD_INPUT_DIR, 'ConGD_phase_2', split, class_folder, vid_name + '.M.avi')
        
        if not os.path.exists(avi_path):
            print(f"Not found: {avi_path}")
            continue

        full_seq = extract_bimanual_sequence(avi_path)  # (T,42,3)

        # 저장할 폴더
        os.makedirs(CONGD_OUTPUT_DIR, exist_ok=True)

        # 각 세그먼트별로 잘라서 npy로 저장
        for start, end, label in seg_list:
            # train.txt는 1-based 인덱스이므로 -1
            seg_seq = full_seq[start-1 : end]  # shape (L,42,3)
            
            # (1) 완전히 0인 시퀀스는 무시
            if np.all(seg_seq == 0):
                print(f"Skip {video_id} (all-zero segment)")
                continue
            
            # (2) 길이가 너무 짧은 시퀀스는 무시
            if seg_seq.shape[0] < FRAMES - 5:
                print(f"Skip {video_id} (too short segment: {seg_seq.shape[0]} frames)")
                continue    
            # 길이를 37에 맞춰 자르거나 패딩하기
            seg_seq = pad_or_trim(seg_seq, FRAMES)

            
            if compute_joint_movement_variance(seg_seq) < 0.001:
                print(f"Skip {video_id} (low movement variance)")
                continue

            # (3) 0이 하나라도 있으면 보간
            if np.any(seg_seq == 0):
                 # (3.0) 0 비율이 너무 많으면 제거
                zero_mask = (seg_seq == 0).all(axis=2)  # shape: (T, 42)
                zero_ratio = zero_mask.sum() / (FRAMES * 42)
                if zero_ratio > 0.3:
                    print(f"Skip {video_id} (too many missing joints: {zero_ratio:.2f})")
                    continue
                seg_seq = interpolate_missing_frames(seg_seq)

            # (4) 저장
            out_name = f"{split}_{class_folder}_{vid_name}_{start:03d}-{end:03d}_lbl{label}.npy"
            out_path = os.path.join(CONGD_OUTPUT_DIR, out_name)
            np.save(out_path, seg_seq)
            print(f"Saved: {out_path} → {seg_seq.shape}, label={label}")

if __name__ == "__main__":
    process_congd()
    # hands.close()
    hand_landmarker.close()

