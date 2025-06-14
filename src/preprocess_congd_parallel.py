import os
import glob
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from variables import *
from utility import pad_or_trim, interpolate_missing_frames

#— worker 함수: 하나의 비디오를 처리 ------------------------
def process_video(args):
    split, class_folder, vid_name, seg_list = args

    # MediaPipe Hands 객체는 각 프로세스에서 따로 생성
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=1
    )

    # 비디오 파일 경로 결정
    if split == 'train':
        avi_path = os.path.join(CONGD_INPUT_DIR, 'ConGD_phase_1', split, class_folder, vid_name + '.M.avi')
    else:
        avi_path = os.path.join(CONGD_INPUT_DIR, 'ConGD_phase_2', split, class_folder, vid_name + '.M.avi')
    if not os.path.exists(avi_path):
        return f"[WARN] Not found: {avi_path}"

    # 전체 시퀀스 추출
    cap = cv2.VideoCapture(avi_path)
    full_seq = []
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
        full_seq.append(left + right)
    cap.release()
    hands.close()
    full_seq = np.array(full_seq, dtype=np.float32)

    # 세그먼트별로 자르고 저장
    os.makedirs(CONGD_OUTPUT_DIR, exist_ok=True)
    out_logs = []
    for start, end, label in seg_list:
        seg_seq = full_seq[start-1 : end]  # 1-based → 0-based
        if np.all(seg_seq == 0) or seg_seq.shape[0] < FRAMES - 5:
            continue
        seg_seq = pad_or_trim(seg_seq, FRAMES)
        if np.any(seg_seq == 0):
            seg_seq = interpolate_missing_frames(seg_seq)

        out_name = f"{split}_{class_folder}_{vid_name}_{start:03d}-{end:03d}_lbl{label}.npy"
        out_path = os.path.join(CONGD_OUTPUT_DIR, out_name)
        # np.save(out_path, seg_seq)
        out_logs.append(f"Saved: {out_path} → {seg_seq.shape}, label={label}")

    return "\n".join(out_logs)


#— master 함수: 전체 세그먼트 맵을 만든 뒤 병렬 실행 ------------
def process_congd_parallel(max_workers=4):
    # phase1/phase2 세그먼트 로드
    def load_segments(txt_path, split):
        segs = defaultdict(list)
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                vid = line.split()[0]
                key = (split, *vid.split('/'))  # (split, class_folder, vid_name)
                for seg in line.strip().split()[1:]:
                    span, lbl = seg.split(':')
                    s, e = map(int, span.split(','))
                    segs[key].append((s, e, int(lbl)))
        return segs

    p1 = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_1", "train.txt")
    p2 = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_2", "valid.txt")
    segs1 = load_segments(p1, "train")
    segs2 = load_segments(p2, "test")
    all_segments = {**segs1, **segs2}

    # 작업 리스트 준비
    tasks = [(split, cls, vid, seg_list)
             for (split, cls, vid), seg_list in all_segments.items()]

    # 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_video, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            print(result)


if __name__ == "__main__":
    process_congd_parallel(max_workers=4)
