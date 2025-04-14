import os
import glob
import cv2
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from variables import train_dir, train_subset_dir, preprocessed_train_dir, preprocessed_train_subset_dir
import logging
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 경고 필터링
warnings.filterwarnings('ignore')

# 로깅 설정 - 파일에도 저장
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# mediapipe 로깅 억제
os.environ["GLOG_minloglevel"] = "3"  # 중요 경고 및 오류만 표시
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 텐서플로우 로그 레벨 (2=경고만)

# # variables.py에서 가져오는 변수들을 직접 정의 (실제 사용 시 from variables import 문을 사용하세요)
# train_dir = "path/to/train"  # 수정 필요
# train_subset_dir = "path/to/train_subset"  # 수정 필요
# preprocessed_train_dir = "path/to/preprocessed_train"  # 수정 필요
# preprocessed_train_subset_dir = "path/to/preprocessed_train_subset"  # 수정 필요

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
            filtered_ts = np.convolve(ts, kernel, mode='same')
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
    
    # 첫 번째와 마지막 프레임이 유효하지 않은 경우, 가장 가까운 유효한 프레임 값으로 채움
    if 0 not in valid_indices:
        new_seq[0] = new_seq[valid_indices[0]]
    if n_frames - 1 not in valid_indices:
        new_seq[n_frames - 1] = new_seq[valid_indices[-1]]
    
    # 유효한 첫 번째와 마지막 프레임을 포함하도록 valid_indices 갱신
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
    # 두 가지 가능한 형식을 모두 시도
    frame_pattern = os.path.join(sample_dir, "*.jpg")
    frame_files = glob.glob(frame_pattern)
    
    if not frame_files:
        # 다른 가능한 이미지 확장자 시도
        for ext in ['png', 'jpeg']:
            frame_pattern = os.path.join(sample_dir, f"*.{ext}")
            frame_files = glob.glob(frame_pattern)
            if frame_files:
                break
    
    if not frame_files:
        logger.warning(f"No frame files found in {sample_dir}")
        return []
    
    # 숫자로 정렬 (파일명에서 숫자 추출)
    def extract_number(filename):
        try:
            # 파일명에서 숫자만 추출
            return int(''.join(filter(str.isdigit, os.path.basename(filename))))
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
        # 원래 stdout을 저장
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # mediapipe 출력 억제를 위해 임시 파일로 리다이렉션
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            
            try:
                # 필요할 때만 mediapipe 임포트
                import mediapipe as mp
                
                frame_files = get_frame_files(sample_dir)
                if not frame_files:
                    return np.array([])
                
                frame_landmarks = []
                
                # mediapipe Hands 초기화
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
                    # 자원 해제
                    hands.close()
                
                return np.array(frame_landmarks)
            
            except ImportError:
                logger.error("Failed to import mediapipe. Please install it with 'pip install mediapipe'")
                return np.array([])
            except Exception:
                return np.array([])
    finally:
        # stdout 복원
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
        
        # 결과 저장 (필요한 경우)
        if output_dir and skeleton_sequence_smoothed is not None:
            save_path = os.path.join(output_dir, f"{sample_id}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, skeleton_sequence_smoothed)
        
        return sample_id, True
    
    except Exception:
        return os.path.basename(sample_path), False

def init_worker():
    """작업자 프로세스 초기화 함수"""
    # 로깅 억제
    os.environ["GLOG_minloglevel"] = "3"  
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    
    # 경고 무시
    warnings.filterwarnings('ignore')
    
    # Python 경고 필터링
    if hasattr(sys, 'warnoptions'):
        for warning in sys.warnoptions:
            warnings.simplefilter("ignore")

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
    
    # 샘플 디렉토리 목록 가져오기
    sample_dirs = sorted([p for p in glob.glob(os.path.join(jester_root_dir, '*')) if os.path.isdir(p)])
    logger.info(f"Found {len(sample_dirs)} sample directories")
    
    if not sample_dirs:
        logger.warning(f"No sample directories found in {jester_root_dir}")
        return []
    
    # 작업자 수 결정 (None이면 CPU 코어 수 사용)
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 배치 처리 준비
    processed_ids = []
    successful_count = 0
    batches = [sample_dirs[i:i + batch_size] for i in range(0, len(sample_dirs), batch_size)]
    
    # 비표준 출력을 파일로 리다이렉션
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} samples)")
        
        # 입력 인자 튜플 생성
        tasks = [(sample_path, output_dir) for sample_path in batch]
        
        # tqdm로 진행률 표시 (file=sys.stdout으로 명시적 지정)
        with tqdm(total=len(tasks), desc=f"Batch {batch_idx+1}", file=sys.stdout) as pbar:
            # ProcessPoolExecutor 초기화
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
                # 프로세스 맵 실행
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
    
    # 샘플 디렉토리 목록 가져오기
    sample_dirs = sorted([p for p in glob.glob(os.path.join(jester_root_dir, '*')) if os.path.isdir(p)])
    logger.info(f"Found {len(sample_dirs)} sample directories")
    
    if not sample_dirs:
        logger.warning(f"No sample directories found in {jester_root_dir}")
        return []
    
    processed_ids = []
    successful_count = 0
    
    # tqdm으로 진행률 표시 (file=sys.stdout으로 명시적 지정)
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

    import logging
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

if __name__ == '__main__':
    # 로깅 메시지가 tqdm과 충돌하지 않도록 설정
    mp_logger = multiprocessing.log_to_stderr()
    mp_logger.setLevel(logging.WARNING)
    
    # 기본 설정으로 실행
    main(num_workers=4)