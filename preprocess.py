import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from variables import train_dir

def moving_average_filter(seq, window_size=3):
    """
    seq: numpy array of shape [n_frames, 21, 3]
    window_size: 이동 평균 윈도우 크기 (홀수 권장)
    """
    new_seq = np.copy(seq)
    for joint in range(seq.shape[1]):
        for coord in range(seq.shape[2]):
            ts = seq[:, joint, coord]
            kernel = np.ones(window_size) / window_size
            filtered_ts = np.convolve(ts, kernel, mode='same')
            new_seq[:, joint, coord] = filtered_ts
    return new_seq

def interpolate_missing_frames(seq):
    """
    seq: numpy array of shape [n_frames, 21, 3]
    누락된(모든 값이 0인) 프레임들을 선형 보간으로 채움
    """
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

def extract_skeleton_sequence_from_jester(sample_dir):
    """
    sample_dir: 특정 샘플 폴더 (frame0001.jpg ~ frame0037.jpg 포함)
    Returns: numpy array of shape [37, 21, 3]
    """
    # 각 자식 프로세스 내에서 mediapipe를 임포트(중복 초기화 방지)
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    # 각 프로세스에서 Hands 객체를 독립적으로 생성
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )
    
    frame_landmarks = []
    for i in range(1, 38):  # 1부터 37까지
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
    hands.close()  # 사용이 끝난 후 자원 해제
    return np.array(frame_landmarks)

def process_sample(sample_path):
    """
    단일 샘플 폴더를 처리하는 함수.
    이미지에서 손 스켈레톤 시퀀스를 추출, 보간 및 필터링하고 npy 파일로 저장.
    """
    sample_id = os.path.basename(sample_path)
    # 원본 시퀀스
    skeleton_sequence = extract_skeleton_sequence_from_jester(sample_path)
    # 보간: 누락된 프레임 채우기
    skeleton_sequence_interp = interpolate_missing_frames(skeleton_sequence)
    # 필터링: 이동 평균 필터 적용해 노이즈 감소
    skeleton_sequence_smoothed = moving_average_filter(skeleton_sequence_interp, window_size=3)
    
    return sample_id, skeleton_sequence_smoothed

def process_jester_dataset_parallel(jester_root_dir, output_dir, num_workers=4):
    """
    jester_root_dir: 각 샘플 폴더(예: '00001/', '00002/', ...)가 있는 루트 디렉토리
    output_dir: 결과 npy 파일을 저장할 디렉토리
    num_workers: 병렬 처리에 사용할 프로세스 수
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dirs = sorted([p for p in glob.glob(os.path.join(jester_root_dir, '*')) if os.path.isdir(p)])
    
    with Pool(num_workers) as pool:
        for sample_id, skeleton_seq in tqdm(pool.imap_unordered(process_sample, sample_dirs), total=len(sample_dirs)):
            np.save(os.path.join(output_dir, f"{sample_id}.npy"), skeleton_seq)

if __name__ == '__main__':
    # 병렬 처리 이전에 부모 프로세스에서는 GPU 관련 라이브러리를 임포트하지 않도록 주의합니다.
    jester_root_dir = "your_jester_dataset_path"
    output_dir = "your_output_directory"
    # 프로세스 수는 시스템 코어 개수에 맞게 설정하는 것이 좋습니다.
    process_jester_dataset_parallel(jester_root_dir, output_dir, num_workers=4)
