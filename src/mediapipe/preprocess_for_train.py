import os
import json
from datetime import datetime
from time import perf_counter

# NOTE: 이 스크립트가 실행되려면 src.mediapipe 모듈과 그 하위 파일들이 필요합니다.
from src.mediapipe.config import FRAME_OUTPUT_DIR
from src.mediapipe.utils import extract_frames_from_video
# NOTE: 효율적인 무결성 검사를 위해 src/mediapipe/utils.py에 get_video_frame_count(video_path) 함수가 필요합니다.
from src.mediapipe.utils import get_video_frame_count 
from src.mediapipe.hand_tracking import HandTracker

# --- 파일 경로 설정 ---
VIDEO_LIST_PATH = "dataset/video_list.json"
PROCESSED_VIDEOS_PATH = "dataset/processed/processed_videos.txt"
FAILED_VIDEOS_PATH = "dataset/processed/failed_videos.txt"
EXTRACTED_VIDEOS_PATH = "dataset/processed/extracted_videos.txt"
PROCESS_LOG_PATH = "dataset/processed/process_log.txt"

# --- 로깅 함수 ---
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(PROCESS_LOG_PATH, 'a') as f:
        f.write(log_line + "\n")

# --- 유틸리티 함수 ---
def save_video_list(root_video_dir, save_path=VIDEO_LIST_PATH):
    """주어진 디렉토리에서 모든 MP4/AVI 파일을 찾아 목록을 JSON으로 저장합니다."""
    video_list = []
    subfolders = sorted([f for f in os.listdir(root_video_dir) if os.path.isdir(os.path.join(root_video_dir, f))])
    for subfolder in subfolders:
        folder_path = os.path.join(root_video_dir, subfolder)
        # Hidden files or system files might be included, ensure they are video files
        video_files = sorted([f for f in os.listdir(folder_path) if (f.endswith(".mp4") or f.endswith(".avi")) and not f.startswith('.')])
        for vf in video_files:
            video_list.append(os.path.join(subfolder, vf))

    with open(save_path, 'w') as f:
        json.dump(video_list, f, indent=2)
    log_message(f"Updated video list with {len(video_list)} videos.")

def load_set_from_file(path):
    """파일에서 항목 목록을 로드하여 Set 형태로 반환합니다."""
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def append_line_to_file(path, line):
    """파일에 새 줄을 추가합니다."""
    with open(path, 'a') as f:
        f.write(line + '\n')

def process_videos(root_video_dir, draw_img_base_dir, landmarks_base_dir, max_count=None, reprocess_mode=False):
    """
    비디오 처리 및 랜드마크 추출을 수행합니다.
    reprocess_mode=True일 경우, 'processed'로 기록되었지만 NPY 파일이 누락되었거나 개수가 부족한 비디오를 재처리합니다.
    """
    hand_tracker = HandTracker()

    # NPY 파일 개수를 세는 헬퍼 함수
    def count_npy_files(directory):
        if not os.path.isdir(directory):
            return 0
        return len([f for f in os.listdir(directory) if f.endswith('.npy')])
    
    # 최신 영상 목록 항상 갱신
    save_video_list(root_video_dir)

    processed_videos = load_set_from_file(PROCESSED_VIDEOS_PATH)
    failed_videos = load_set_from_file(FAILED_VIDEOS_PATH)
    extracted_videos = load_set_from_file(EXTRACTED_VIDEOS_PATH)

    with open(VIDEO_LIST_PATH, 'r') as f:
        video_list = json.load(f)

    count = 0
    total_videos = len(video_list)
    start_time = perf_counter()

    mode_info = "REPROCESS_MISSING_NPY_OR_INCOMPLETE Mode (FULL CHECK)" if reprocess_mode else "NORMAL Processing Mode"
    log_message(f"Starting processing ({mode_info}). Total videos listed: {total_videos}")

    # --- 반복 처리 시작 ---
    for idx, video_rel_path in enumerate(video_list):
        subfolder, video_file = os.path.split(video_rel_path)
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(root_video_dir, subfolder, video_file)
        
        # 랜드마크 저장 경로
        landmarks_save_dir = os.path.join(landmarks_base_dir, subfolder, video_id)
        
        # 처리 상태 확인
        video_is_processed = video_rel_path in processed_videos
        
        # 실제 NPY 파일 개수
        actual_npy_count = count_npy_files(landmarks_save_dir)
        npy_files_exist = actual_npy_count > 0 

        # 1. 실패한 영상은 건너뜁니다.
        if video_rel_path in failed_videos:
            log_message(f"--- SKIPPING: {video_rel_path} - Previously FAILED.")
            continue

        # 2. 처리 건너뛰기/재처리 로직
        
        # A. Normal Mode: 이미 처리 목록에 있으면 NPY 존재 여부와 관계없이 무조건 건너뜀 (기존 동작)
        if not reprocess_mode and video_is_processed:
            continue
            
        # B. Reprocess Mode 또는 미처리된 영상에 대해서만 정밀 체크 시작
        
        # B-1. Reprocess Mode에서 이미 처리된 파일에 대한 무결성 검사
        if reprocess_mode and video_is_processed:
            
            # --- (1) 예상 프레임 수 확인 (Utils의 frame_count 함수가 필요) ---
            expected_frame_count = -1
            try:
                # get_video_frame_count는 src.mediapipe.utils에 구현되어야 함
                from src.mediapipe.utils import get_video_frame_count 
                expected_frame_count = get_video_frame_count(video_path)
            except (ImportError, AttributeError) as e:
                log_message(f"WARNING: Cannot determine expected frame count for {video_rel_path}. Check src.mediapipe.utils.get_video_frame_count: {repr(e)}")
            
            # --- (2) 무결성 검사 ---
            if expected_frame_count > 0 and actual_npy_count == expected_frame_count:
                # 랜드마크 파일 개수가 예상 프레임 개수와 정확히 일치하면 건너뜀
                log_message(f"--- SKIPPING: {video_rel_path} - PROCESSED (Count OK: {actual_npy_count}/{expected_frame_count}).")
                continue
            
            # --- (3) 재처리 필요 조건 ---
            reprocess_reason = None
            if not npy_files_exist:
                reprocess_reason = "MISSING NPY files"
            elif expected_frame_count > 0 and actual_npy_count > 0 and actual_npy_count < expected_frame_count:
                reprocess_reason = f"INCOMPLETE NPY files ({actual_npy_count}/{expected_frame_count})"
            
            if reprocess_reason:
                log_message(f"--- REPROCESSING REQUIRED: {video_rel_path} - Reason: {reprocess_reason}. Proceeding to re-extract.")
            # 재처리 필요하지 않은 모든 케이스는 continue로 처리되었으므로,
            # 여기에 도달한 video_is_processed=True 파일은 재처리됨.
            
        # 3. 비디오 처리 시작
        log_message(f"[{idx+1}/{total_videos}] Processing video: {video_rel_path}")

        try:
            frame_output_dir = FRAME_OUTPUT_DIR
            draw_img_save_dir = os.path.join(draw_img_base_dir, subfolder, video_id)
            
            # 3.1. 랜드마크 저장 디렉토리 생성/확인
            os.makedirs(landmarks_save_dir, exist_ok=True)
            
            # 3.2. 프레임 추출 (최초 1회만)
            # 재처리 모드에서는 정확한 프레임 파일을 확보하기 위해 extracted_videos 목록을 무시하고 추출을 다시 시도합니다.
            if video_rel_path not in extracted_videos or reprocess_mode:
                log_message(f"Extracting frames from video...")
                frame_count = extract_frames_from_video(video_path, frame_output_dir)
                log_message(f"Extracted {frame_count} frames.")
                # 재처리 모드에서는 목록에 이미 있다면 다시 추가하지 않음
                if video_rel_path not in extracted_videos:
                    append_line_to_file(EXTRACTED_VIDEOS_PATH, video_rel_path)
            else:
                log_message(f"Frames already extracted for this video, skipping extraction.")

            # 3.3. 프레임 처리 및 랜드마크 저장
            frame_files = sorted([f for f in os.listdir(frame_output_dir) if f.endswith(".jpg")])
            if not frame_files:
                # 프레임 추출이 실패했거나 FRAME_OUTPUT_DIR이 비어있는 경우
                raise FileNotFoundError(f"No frames found in temporary directory: {frame_output_dir}")

            for frame_file in frame_files:
                frame_path = os.path.join(frame_output_dir, frame_file)
                landmarks = hand_tracker.process_image(frame_path)
                basename = os.path.splitext(frame_file)[0]
                save_lm_path = os.path.join(landmarks_save_dir, f"{basename}_landmarks.npy")
                hand_tracker.save_landmarks(landmarks, save_lm_path)
            
            # 3.4. 성공 기록
            if not video_is_processed:
                append_line_to_file(PROCESSED_VIDEOS_PATH, video_rel_path)
                log_message(f"Successfully processed and saved landmarks for {video_rel_path}")
            else:
                log_message(f"Successfully REPROCESSED and saved landmarks for {video_rel_path}")


            count += 1
            if max_count and count >= max_count:
                log_message(f"Reached max_count {max_count}, stopping batch processing.")
                break

        except Exception as e:
            # 3.5. 오류 발생 시 실패 기록
            log_message(f"Error processing video {video_rel_path}: {repr(e)}")
            
            # 이미 FAILED_VIDEOS에 없다면 추가
            if video_rel_path not in failed_videos:
                append_line_to_file(FAILED_VIDEOS_PATH, video_rel_path)
            
            continue

    end_time = perf_counter()
    elapsed = end_time - start_time
    log_message(f"Processing session finished. Videos processed this run: {count}")
    log_message(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    hand_tracker.close()