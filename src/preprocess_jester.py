# extract mediapipe features from the dataset
import os, glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from variables import *
from utility import hands, \
                    pad_or_trim, interpolate_missing_frames

# 37 frames에서 (T,42,3) 전체 시퀀스 뽑기
def extract_bimanual_sequence(frames_dir):
    """
    Args:
        frames_dir: JPG 시퀀스가 들어있는 폴더 경로
    Returns:
        ndarray of shape (T, 42, 3), dtype float32
        — 각 프레임마다 [L0...L20, R0...R20] 순
    """
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    seq = []

    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            # 이미지 로드 실패 시 양손 모두 0으로
            coords = [(0.0,0.0,0.0)] * 42
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # 기본값: 양손 21점씩 0으로
            left = [(0.0,0.0,0.0)] * 21
            right= [(0.0,0.0,0.0)] * 21

            if res.multi_hand_landmarks and res.multi_handedness:
                # multi_handedness 로 왼/오 구분
                for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    coords21 = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    if label == "Left":
                        left = coords21
                    else:
                        right = coords21

            coords = left + right
        seq.append(coords)
    return np.array(seq, dtype=np.float32)  # (T, 42, 3)

# frames to skeletons
def process_jester_skeletons():
    """
    폴더 구조:
    raw_data/
      ├─ Train/
      │    ├─ <sample1>/
      │    │    ├─ 00001.jpg
      │    │    ├─ 00002.jpg
      │    │    └─ …
      │    └─ <sample2>/ …
      ├─ Validation/ …
      └─ Test/ …
    """
    for split in ("Train", "Validation", "Test"):
        in_dir  = os.path.join(JESTER_INPUT_DIR, split)
        out_dir = os.path.join(JESTER_OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        for sample in sorted(os.listdir(in_dir)):
            print(f"Processing {sample}...")
            samp_dir = os.path.join(in_dir, sample)
            if not os.path.isdir(samp_dir):
                continue

            arr = extract_bimanual_sequence(samp_dir)
            # (1) 완전히 0인 시퀀스이고 no-gesture 가 아니라면 skip
            all_zero = np.all(arr == 0)
            label_id = mapping.get(int(sample), None)       # sample → label_id 매핑
            label = JESTER_LABELS.get(label_id, None)       # label_id → label_name 매핑

            if all_zero and label != 'No gesture':
                print(f"Skip {sample} (all-zero & label={label})")
                continue

            # (2) 길이 맞추기
            if arr.shape[0] != FRAMES:
                arr = pad_or_trim(arr, FRAMES)
    
            # (3) 0이 하나라도 있으면 보간
            if np.any(arr == 0):
                arr = interpolate_missing_frames(arr)

            # (4) 저장
            out_path = os.path.join(out_dir, f"{sample}_lbl{label}.npy")
            # np.save(out_path, arr)
            print(f"Saved {out_path} → {arr.shape}, label={label}")

if __name__ == "__main__":
    train_csv = os.path.join(JESTER_INPUT_DIR, "Train.csv")
    val_csv   = os.path.join(JESTER_INPUT_DIR, "Validation.csv")

    # 1) 데이터프레임 로드
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    # 2) 합치기
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    # 3) video_id → label_id 매핑
    mapping = (
        df_all
        .drop_duplicates(subset="video_id")      # 각 비디오 한 번만
        .set_index("video_id")["label_id"]       # 인덱스를 video_id로, 값은 label_id
        .to_dict()
    )

    # # 확인
    # print("video → id:", list(mapping.items())[:5])
    # print("id → name:", list(mapping_id_to_name.items())[:5])

    # Process Jester dataset
    process_jester_skeletons()
    hands.close()

   



