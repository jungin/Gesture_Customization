import os
import cv2
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
import mediapipe as mp

import av

from src.variables import CONGD_INPUT_DIR, CONGD_OUTPUT_DIR, FRAMES, MP_HANDS_MODEL
from src.utility import pad_or_trim, compute_joint_movement_variance, interpolate_missing_frames

# 1) HandLandmarker 초기화 (글로벌)
base_options = python.BaseOptions(
    model_asset_path=MP_HANDS_MODEL,
    delegate=python.BaseOptions.Delegate.GPU
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE  # ← 여기만 IMAGE로
)
_global_landmarker = vision.HandLandmarker.create_from_options(options)

def load_segments(txt_path, split):
    segs = defaultdict(list)
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            vid = line.split()[0]
            key = (split, *vid.split('/'))
            for seg in line.strip().split()[1:]:
                span, lbl = seg.split(':')
                s_str, e_str = span.split(',')
                segs[key].append((int(s_str), int(e_str), int(lbl)))
    return segs


import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 쓰레드 로컬 캐시
thread_local = threading.local()
def get_landmarker():
    if not hasattr(thread_local, "lm"):
        # IMAGE 모드로만 초기화
        base_options = python.BaseOptions(
            model_asset_path=MP_HANDS_MODEL,
            delegate=python.BaseOptions.Delegate.GPU
        )
        opts = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE   # ← IMAGE 모드!
        )
        thread_local.lm = vision.HandLandmarker.create_from_options(opts)
    return thread_local.lm


def extract_bimanual_sequence(video_path):
    container = av.open(video_path)
    stream    = container.streams.video[0]
    fps       = float(stream.average_rate) if stream.average_rate else 10.0
    ms_per_frame = 1000.0 / fps

    seq       = []
    lm        = get_landmarker()   # 이제 IMAGE 모드 인스턴스
    frame_idx = 0

    for packet in container.demux(video=0):
        for frame in packet.decode():
            rgb = frame.to_ndarray(format='rgb24')
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            mp_img = Image(mp.ImageFormat.SRGB, rgb)

            # ▶︎ detect_for_video 대신 detect() 호출
            result = lm.detect(mp_img)

            left  = [(0.,0.,0.)]*21
            right = [(0.,0.,0.)]*21
            if result.hand_landmarks and result.handedness:
                for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                    coords = [(p.x,p.y,p.z) for p in landmarks]
                    if handed[0].category_name == "Left":
                        left = coords
                    else:
                        right = coords

            seq.append(left + right)
            frame_idx += 1

    container.close()
    return np.array(seq, dtype=np.float32)


def process_one_video(args):
    split, cls, vid, seg_list = args
    # 비디오 경로 구성
    if split == "train":
        avi = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_1", split, cls, vid + ".M.avi")
    else:
        avi = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_2", split, cls, vid + ".M.avi")
    if not os.path.exists(avi):
        return f"[WARN] Not found: {avi}"

    full_seq = extract_bimanual_sequence(avi)

    os.makedirs(CONGD_OUTPUT_DIR, exist_ok=True)
    logs = []
    for s, e, lbl in seg_list:
        seg = full_seq[s-1:e]
        if seg.shape[0] < FRAMES - 5:
            continue
        seg = pad_or_trim(seg, FRAMES)
        if compute_joint_movement_variance(seg) < 0.001:
            continue
        if np.any(seg == 0):
            # (3.0) 0 비율이 너무 많으면 제거
            zero_mask = (seg == 0).all(axis=2)  # shape: (T, 42)
            zero_ratio = zero_mask.sum() / (FRAMES * 42)
            if zero_ratio > 0.3:
                continue
            seg = interpolate_missing_frames(seg)

        name = f"{split}_{cls}_{vid}_{s:03d}-{e:03d}_lbl{lbl}.npy"
        path = os.path.join(CONGD_OUTPUT_DIR, name)
        np.save(path, seg)
        logs.append(f"Saved {name}")
    return "\n".join(logs)

def process_congd_parallel():
    # 세그먼트 로드
    segs1 = load_segments(os.path.join(CONGD_INPUT_DIR, "ConGD_phase_1", "train.txt"), "train")
    segs2 = load_segments(os.path.join(CONGD_INPUT_DIR, "ConGD_phase_2", "valid.txt"), "test")
    all_segs = {**segs1, **segs2}

    # 작업 리스트 생성
    tasks = [
        (split, cls, vid, seg_list)
        for (split, cls, vid), seg_list in all_segs.items()
    ]

    # ThreadPoolExecutor로 병렬 처리
    with ThreadPoolExecutor(max_workers=4) as exe:
        futures = {exe.submit(process_one_video, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result().strip()    # 앞뒤 공백 제거
            if res:                       # 비어 있지 않을 때만 출력
                print(res)
if __name__ == "__main__":
    process_congd_parallel()
    _global_landmarker.close()

    
# import os
# import glob
# import numpy as np
# import cv2
# import mediapipe as mp
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor, as_completed

# from variables import *
# from utility import pad_or_trim, interpolate_missing_frames


# # phase1/phase2 세그먼트 로드
# def load_segments(txt_path, split):
#     segs = defaultdict(list)
#     with open(txt_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             vid = line.split()[0]
#             key = (split, *vid.split('/'))
#             for seg in line.strip().split()[1:]:
#                 span, lbl = seg.split(':')
#                 s, e = map(int, span.split(','))
#                 segs[key].append((s, e, int(lbl)))
#     return segs

# #— worker 함수: 하나의 비디오를 처리 ------------------------
# def process_video(args):
#     try:
#         split, class_folder, vid_name, seg_list = args

#         # MediaPipe Hands 객체는 각 프로세스에서 따로 생성
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.5,
#             model_complexity=1
#         )

#         # 비디오 파일 경로 결정
#         if split == 'train':
#             avi_path = os.path.join(CONGD_INPUT_DIR, 'ConGD_phase_1', split, class_folder, vid_name + '.M.avi')
#         else:
#             avi_path = os.path.join(CONGD_INPUT_DIR, 'ConGD_phase_2', split, class_folder, vid_name + '.M.avi')
#         if not os.path.exists(avi_path):
#             return f"[WARN] Not found: {avi_path}"

#         # 전체 시퀀스 추출
#         cap = cv2.VideoCapture(avi_path)
#         full_seq = []
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = hands.process(rgb)

#             left  = [(0.,0.,0.)]*21
#             right = [(0.,0.,0.)]*21
#             if res.multi_hand_landmarks and res.multi_handedness:
#                 for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
#                     coords21 = [(p.x, p.y, p.z) for p in lm.landmark]
#                     if hd.classification[0].label == "Left":
#                         left = coords21
#                     else:
#                         right = coords21
#             full_seq.append(left + right)
#         cap.release()
#         hands.close()
#         full_seq = np.array(full_seq, dtype=np.float32)

#         # 세그먼트별로 자르고 저장
#         os.makedirs(CONGD_OUTPUT_DIR, exist_ok=True)
#         out_logs = []
#         for start, end, label in seg_list:
#             seg_seq = full_seq[start-1 : end]  # 1-based → 0-based
#             if np.all(seg_seq == 0) or seg_seq.shape[0] < FRAMES - 5:
#                 continue
#             seg_seq = pad_or_trim(seg_seq, FRAMES)
#             if np.any(seg_seq == 0):
#                 seg_seq = interpolate_missing_frames(seg_seq)

#             out_name = f"{split}_{class_folder}_{vid_name}_{start:03d}-{end:03d}_lbl{label}.npy"
#             out_path = os.path.join(CONGD_OUTPUT_DIR, out_name)
#             np.save(out_path, seg_seq)
#             out_logs.append(f"Saved: {out_path} → {seg_seq.shape}, label={label}")

#         return "\n".join(out_logs)
#     except Exception as e:
#         import traceback, sys
#         traceback.print_exc(file=sys.stderr)
#         return f"[ERROR] {args} → {e}"


# if __name__ == "__main__":
#     # 세그먼트 로딩
#     p1 = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_1", "train.txt")
#     p2 = os.path.join(CONGD_INPUT_DIR, "ConGD_phase_2", "valid.txt")
#     segs1 = load_segments(p1, "train")
#     segs2 = load_segments(p2, "test")
#     all_segments = {**segs1, **segs2}

#     # 작업 리스트 준비
#     tasks = [(split, cls, vid, seg_list)
#              for (split, cls, vid), seg_list in all_segments.items()]

#     # 프로세스 풀 실행
#     with ProcessPoolExecutor(max_workers=4) as exe:
#         futures = {exe.submit(process_video, t): t for t in tasks}
#         for fut in as_completed(futures):
#             try:
#                 print(fut.result())
#             except Exception as e:
#                 print(f"[ERROR] {futures[fut]} → {e}")

