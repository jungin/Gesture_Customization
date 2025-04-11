import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import glob
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from variables import train_dir, subset_dir, preprocessed_train_dir

def process_sample(sample_path_and_output):
    # 각 자식 프로세스에서는 여기서 mediapipe 등 GPU 관련 라이브러리를 임포트합니다.
    from contextlib import redirect_stderr

    # mediapipe 등 GPU 관련 라이브러리의 임포트 및 초기화를 위해 출력 억제
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        import mediapipe as mp

    sample_path, output_dir = sample_path_and_output
    sample_id = os.path.basename(sample_path)

    # 각 프로세스별로 mediapipe Hands 객체를 독립적으로 생성합니다.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=1,
                           min_detection_confidence=0.5)

    def extract_skeleton_sequence(sample_dir):
        frame_landmarks = []
        for i in range(1, 38):  # 1부터 37까지 프레임 처리
            img_path = os.path.join(sample_dir, f"{i:05d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image not found: {img_path}")
                frame_landmarks.append(np.zeros((21, 3)))
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            else:
                landmarks = np.zeros((21, 3))
            frame_landmarks.append(landmarks)
        return np.array(frame_landmarks)

    def interpolate_missing_frames(seq):
        n_frames = seq.shape[0]
        frame_indices = np.arange(n_frames)
        new_seq = seq.copy()
        valid_indices = [i for i in range(n_frames) if not np.all(seq[i] == 0)]
        if len(valid_indices) == 0:
            return new_seq
        for joint in range(seq.shape[1]):
            for coord in range(seq.shape[2]):
                ts = seq[:, joint, coord]
                x_valid = np.array(valid_indices)
                y_valid = ts[x_valid]
                new_ts = np.interp(frame_indices, x_valid, y_valid)
                new_seq[:, joint, coord] = new_ts
        return new_seq

    def moving_average_filter(seq, window_size=3):
        new_seq = np.copy(seq)
        for joint in range(seq.shape[1]):
            for coord in range(seq.shape[2]):
                ts = seq[:, joint, coord]
                kernel = np.ones(window_size) / window_size
                filtered_ts = np.convolve(ts, kernel, mode='same')
                new_seq[:, joint, coord] = filtered_ts
        return new_seq

    skeleton_sequence = extract_skeleton_sequence(sample_path)
    skeleton_sequence_interp = interpolate_missing_frames(skeleton_sequence)
    skeleton_sequence_smoothed = moving_average_filter(skeleton_sequence_interp, window_size=3)
    np.save(os.path.join(output_dir, f"{sample_id}.npy"), skeleton_sequence_smoothed)

    # 작업 완료 후 자원 해제 (필요하면 hands.close() 호출)
    hands.close()

    return sample_id

def process_jester_dataset_parallel(jester_root_dir, output_dir, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    sample_dirs = sorted([p for p in glob.glob(os.path.join(jester_root_dir, '*')) if os.path.isdir(p)])
    tasks = [(sample_path, output_dir) for sample_path in sample_dirs]
    with Pool(num_workers) as pool:
        for sample_id in tqdm(pool.imap_unordered(process_sample, tasks), total=len(tasks)):
            # 진행 상황을 표시합니다.
            pass

if __name__ == '__main__':
    # 멀티프로세싱 환경에서는 부모 프로세스에서 GPU 관련 라이브러리 임포트를 최소화합니다.
    # 각 자식 프로세스에서 독립적으로 임포트하게 합니다.
    jester_root_dir = "path_to_your_jester_dataset"
    output_dir = "path_to_save_output"

    process_jester_dataset_parallel(train_dir, preprocessed_train_dir, num_workers=4)
    # process_jester_dataset_parallel(jester_root_dir, output_dir, num_workers=4)
