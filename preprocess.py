import os, sys
# 프로그램 시작 시 전체 stderr를 /dev/null로 리다이렉션 (모든 stderr 메시지 숨김)
devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, sys.stderr.fileno())

import glob
import cv2
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from variables import train_dir, train_subset_dir, preprocessed_train_dir, preprocessed_train_subset_dir
import logging
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import warnings
import absl.logging

# --- 환경 변수 및 초기 설정 ---
# absl의 로깅을 FATAL로 설정하여, 정보/경고 메시지를 숨깁니다.
absl.logging.set_verbosity(absl.logging.FATAL)
os.environ["GLOG_minloglevel"] = "3"  # 중요 경고 및 오류만 표시
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 텐서플로우: 오류만

# 경고 필터링
warnings.filterwarnings("ignore")

# --- tqdm와 함께 사용하기 위한 커스텀 로깅 핸들러 ---
class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stdout)
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# --- 로깅 설정: 파일에는 로그 남기고, 콘솔에는 TqdmLoggingHandler 사용 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 기존 핸들러 제거 후 새로운 핸들러 추가
logger.handlers = []
file_handler = logging.FileHandler("preprocessing.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler = TqdmLoggingHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def moving_average_filter(seq, window_size=3):
    """
    시퀀스에 이동 평균 필터 적용
    Args:
        seq: numpy array of shape [n_frames, 21, 3]
        window_size: 이동 평균 윈도우 크기 (홀수 권장)
    Returns:
        필터링된 시퀀스
    """
    if seq.size == 0:
        return seq
    new_seq = np.copy(seq)
    for joint in range(seq.shape[1]):
        for coord in range(seq.shape[2]):
            ts = seq[:, joint, coord]
            kernel = np.ones(window_size) / window_size
            filtered_ts = np.convolve(ts, kernel, mode="same")
            new_seq[:, joint, coord] = filtered_ts
    return new_seq

def interpolate_missing_frames(seq):
    """
    누락된(모든 값이 0인) 프레임들을 선형 보간으로 채움
    Args:
        seq: numpy array of shape [n_frames, 21, 3]
    Returns:
        보간된 시퀀스
    """
    if seq.size == 0:
        return seq
    n_frames = seq.shape[0]
    frame_indices = np.arange(n_frames)
    new_seq = seq.copy()
    valid_indices = [i for i in range(n_frames) if not np.all(seq[i] == 0)]
    if len(valid_indices) == 0:
        logger.warning("No valid frames found for interpolation")
        return new_seq
    if 0 not in valid_indices:
        new_seq[0] = new_seq[valid_indices[0]]
    if n_frames - 1 not in valid_indices:
        new_seq[n_frames - 1] = new_seq[valid_indices[-1]]
    if 0 not in valid_indices:
        valid_indices.insert(0, 0)
    if n_frames - 1 not in valid_indices:
        valid_indices.append(n_frames - 1)
    for joint in range(seq.shape[1]):
        for coord in range(seq.shape[2]):
            ts = new_seq[:, joint, coord]
            x_valid = np.array(valid_indices)
            y_valid = ts[x_valid]
            new_ts = np.interp(frame_indices, x_valid, y_valid)
            new_seq[:, joint, coord] = new_ts
    return new_seq

def get_frame_files(sample_dir):
    """
    샘플 디렉토리에서 프레임 이미지 파일들을 찾아 정렬된 리스트로 반환
    Args:
        sample_dir: 샘플 폴더 경로
    Returns:
        정렬된 이미지 파일 경로 리스트
    """
    frame_pattern = os.path.join(sample_dir, "*.jpg")
    frame_files = glob.glob(frame_pattern)
    if not frame_files:
        for ext in ["png", "jpeg"]:
            frame_pattern = os.path.join(sample_dir, f"*.{ext}")
            frame_files = glob.glob(frame_pattern)
            if frame_files:
                break
    if not frame_files:
        logger.warning(f"No frame files found in {sample_dir}")
        return []
    def extract_number(filename):
        try:
            return int("".join(filter(str.isdigit, os.path.basename(filename))))
        except:
            return 0
    return sorted(frame_files, key=extract_number)

def extract_skeleton_sequence(sample_dir):
    """
    이미지 시퀀스에서 손 스켈레톤 시퀀스 추출
    Args:
        sample_dir: 특정 샘플 폴더
    Returns:
        numpy array of shape [n_frames, 21, 3]
    """
    try:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                import mediapipe as mp
                frame_files = get_frame_files(sample_dir)
                if not frame_files:
                    return np.array([])
                frame_landmarks = []
                hands = mp.solutions.hands.Hands(
                    static_image_mode=True, 
                    max_num_hands=1, 
                    min_detection_confidence=0.5
                )
                try:
                    for img_path in frame_files:
                        try:
                            img = cv2.imread(img_path)
                            if img is None:
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
                        except Exception:
                            frame_landmarks.append(np.zeros((21, 3)))
                finally:
                    hands.close()
                return np.array(frame_landmarks)
            except ImportError:
                logger.error("Failed to import mediapipe. Please install it with 'pip install mediapipe'")
                return np.array([])
            except Exception:
                return np.array([])
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def process_sample(args):
    """
    단일 샘플 폴더를 처리하는 함수
    Args:
        args: (sample_path, output_dir) 튜플
    Returns:
        (sample_id, success) 튜플
    """
    sample_path, output_dir = args
    try:
        sample_id = os.path.basename(sample_path)
        skeleton_sequence = extract_skeleton_sequence(sample_path)
        if skeleton_sequence.size == 0:
            return sample_id, False
        skeleton_sequence_interp = interpolate_missing_frames(skeleton_sequence)
        skeleton_sequence_smoothed = moving_average_filter(skeleton_sequence_interp, window_size=3)
        if output_dir and skeleton_sequence_smoothed is not None:
            save_path = os.path.join(output_dir, f"{sample_id}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, skeleton_sequence_smoothed)
        return sample_id, True
    except Exception:
        return os.path.basename(sample_path), False

def init_worker():
    """작업자 프로세스 초기화 함수"""
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import absl.logging
    absl.logging.set_verbosity(absl.logging.FATAL)
    warnings.filterwarnings("ignore")
    # 중요한: 작업자에서는 stderr를 devnull로 리다이렉션해 mediapipe 등의 네이티브 로그 출력 억제
    sys.stderr = open(os.devnull, "w")
    # 만약 추가적으로 개별 모듈 로거 조정이 필요하면 여기서 설정

def process_jester_dataset_parallel(jester_root_dir, output_dir, batch_size=100, num_workers=None):
    """
    전체 데이터셋을 병렬 처리하는 함수 (ProcessPoolExecutor 사용)
    Args:
        jester_root_dir: 각 샘플 폴더가 있는 루트 디렉토리
        output_dir: 결과 npy 파일을 저장할 디렉토리
        batch_size: 한 번에 처리할 샘플 수
        num_workers: 병렬 작업자 수 (None이면 자동 설정)
    Returns:
        처리된 샘플 ID 목록
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dirs = sorted([p for p in glob.glob(os.path.join(jester_root_dir, "*")) if os.path.isdir(p)])
    logger.info(f"Found {len(sample_dirs)} sample directories")
    if not sample_dirs:
        logger.warning(f"No sample directories found in {jester_root_dir}")
        return []
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    processed_ids = []
    successful_count = 0
    batches = [sample_dirs[i : i + batch_size] for i in range(0, len(sample_dirs), batch_size)]
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} samples)")
        tasks = [(sample_path, output_dir) for sample_path in batch]
        with tqdm(total=len(tasks), desc=f"Batch {batch_idx + 1}", file=sys.stdout) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
                for sample_id, success in executor.map(process_sample, tasks):
                    processed_ids.append(sample_id)
                    if success:
                        successful_count += 1
                    pbar.update(1)
    logger.info(f"Processed {len(processed_ids)} samples, {successful_count} successful")
    return processed_ids

def process_dataset_sequential(jester_root_dir, output_dir):
    """
    순차적으로 데이터셋을 처리하는 함수 (디버깅이나 문제 해결용)
    Args:
        jester_root_dir: 각 샘플 폴더가 있는 루트 디렉토리
        output_dir: 결과 npy 파일을 저장할 디렉토리
    Returns:
        처리된 샘플 ID 목록
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_dirs = sorted([p for p in glob.glob(os.path.join(jester_root_dir, "*")) if os.path.isdir(p)])
    logger.info(f"Found {len(sample_dirs)} sample directories")
    if not sample_dirs:
        logger.warning(f"No sample directories found in {jester_root_dir}")
        return []
    processed_ids = []
    successful_count = 0
    with tqdm(total=len(sample_dirs), desc="Processing samples", file=sys.stdout) as pbar:
        for sample_path in sample_dirs:
            sample_id, success = process_sample((sample_path, output_dir))
            processed_ids.append(sample_id)
            if success:
                successful_count += 1
            pbar.update(1)
    logger.info(f"Processed {len(processed_ids)} samples, {successful_count} successful")
    return processed_ids

def main(jester_root_dir=None, output_dir=None, subset=True, batch_size=100, num_workers=None, use_parallel=True):
    """
    메인 함수
    Args:
        jester_root_dir: 입력 디렉토리 (None이면 variables.py에서 가져옴)
        output_dir: 출력 디렉토리 (None이면 variables.py에서 가져옴)
        subset: True면 subset 사용, False면 전체 데이터셋 사용
        batch_size: 한 번에 처리할 샘플 수
        num_workers: 병렬 작업자 수 (None이면 자동 설정)
        use_parallel: True면 병렬 처리, False면 순차 처리
    """
    logging.getLogger("mediapipe").setLevel(logging.FATAL)
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    if jester_root_dir is None:
        jester_root_dir = train_subset_dir if subset else train_dir
    if output_dir is None:
        output_dir = preprocessed_train_subset_dir if subset else preprocessed_train_dir
    logger.info(f"Processing dataset from {jester_root_dir}")
    logger.info(f"Saving results to {output_dir}")
    if use_parallel:
        process_jester_dataset_parallel(jester_root_dir, output_dir, batch_size, num_workers)
    else:
        process_dataset_sequential(jester_root_dir, output_dir)

if __name__ == "__main__":
    mp_logger = multiprocessing.log_to_stderr()
    mp_logger.setLevel(logging.WARNING)
    main(num_workers=8, subset=False)
