import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict

from variables import *
from utility import hands, \
                    pad_or_trim, interpolate_missing_frames

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
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        left  = [(0.,0.,0.)]*21
        right = [(0.,0.,0.)]*21
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                coords21 = [(p.x, p.y, p.z) for p in lm.landmark]
                if hd.classification[0].label == "Left":
                    left = coords21
                else:
                    right = coords21
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

            # (3) 0이 하나라도 있으면 보간
            if np.any(seg_seq == 0):
                seg_seq = interpolate_missing_frames(seg_seq)

            # (4) 저장
            out_name = f"{split}_{class_folder}_{vid_name}_{start:03d}-{end:03d}_lbl{label}.npy"
            out_path = os.path.join(CONGD_OUTPUT_DIR, out_name)
            # np.save(out_path, seg_seq)
            print(f"Saved: {out_path} → {seg_seq.shape}, label={label}")

if __name__ == "__main__":
    process_congd()
    hands.close()
