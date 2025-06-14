import os, glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed

from variables import *
from utility import pad_or_trim, interpolate_missing_frames
import mediapipe as mp

# --- (1) 비디오 프레임 폴더 하나를 처리하는 워커 함수 ---
def process_sample_jester(split, sample, mapping):
    in_dir  = os.path.join(JESTER_INPUT_DIR, split)
    out_dir = os.path.join(JESTER_OUTPUT_DIR, split)
    samp_dir = os.path.join(in_dir, sample)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(samp_dir):
        return f"[SKIP] Not a dir: {samp_dir}"

    # MediaPipe Hands 객체를 각 워커에서 생성
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=1
    )
    
    # (2) 프레임 시퀀스에서 42점 스켈레톤 뽑기
    frame_paths = sorted(glob.glob(os.path.join(samp_dir, "*.jpg")))
    seq = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            coords = [(0.0,0.0,0.0)] * 42
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            left  = [(0.0,0.0,0.0)] * 21
            right = [(0.0,0.0,0.0)] * 21
            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    pts = [(p.x, p.y, p.z) for p in lm.landmark]
                    if hd.classification[0].label == "Left":
                        left = pts
                    else:
                        right = pts
            coords = left + right
        seq.append(coords)
    hands.close()
    arr = np.array(seq, dtype=np.float32)  # (T, 42, 3)

    # (3) 필터링 & 후처리
    label_id = mapping.get(int(sample), None)
    label = JESTER_LABELS.get(label_id, None)
    if np.all(arr == 0) and label != 'No gesture':
        return f"[SKIP] All-zero & not no-gesture: {sample}"
    if arr.shape[0] != FRAMES:
        arr = pad_or_trim(arr, FRAMES)
    if np.any(arr == 0):
        arr = interpolate_missing_frames(arr)

    # (4) 저장
    out_path = os.path.join(out_dir, f"{sample}_lbl{label}.npy")
    np.save(out_path, arr)
    return f"[SAVED] {out_path} → {arr.shape}, label={label}"

# --- (2) 매인 스크립트: 매핑 만들고 병렬 실행 ---
if __name__ == "__main__":
    # 1) 매핑 로드
    df_train = pd.read_csv(os.path.join(JESTER_INPUT_DIR, "Train.csv"))
    df_val   = pd.read_csv(os.path.join(JESTER_INPUT_DIR, "Validation.csv"))
    mapping = (
        pd.concat([df_train, df_val], ignore_index=True)
          .drop_duplicates(subset="video_id")
          .set_index("video_id")["label_id"]
          .to_dict()
    )

    # 2) 작업 리스트 구성
    splits = ["Train", "Validation", "Test"]
    tasks = []
    for split in splits:
        in_dir = os.path.join(JESTER_INPUT_DIR, split)
        for sample in sorted(os.listdir(in_dir)):
            tasks.append((split, sample, mapping))

    # 3) 병렬 실행 (n_jobs에 원하는 프로세스 수 지정)
    results = Parallel(n_jobs=4, backend="multiprocessing")(
        delayed(process_sample_jester)(split, sample, mapping)
        for split, sample, mapping in tasks
    )

    # 4) 결과 로그 출력
    for r in results:
        print(r)
