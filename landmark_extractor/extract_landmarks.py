"""
MediaPipe 랜드마크 추출
"""

import os
import shutil
from time import perf_counter

from utils import log_message, append_line_to_file, load_set_from_file
from landmark_extractor.config import (
    RAW_DIR,
    FRAME_DIR,
    LANDMARKS_DIR,
    DRAW_IMG_DIR,
    VIDEO_LIST_PATH,
    TOPROCESS_LIST_PATH,
    PROCESSED_LIST_PATH,
    EXTRACTED_LIST_PATH,
    FAILED_VIDEOS_PATH,
)
from landmark_extractor.utils import extract_frames_from_video
from landmark_extractor.hand_tracking import HandTracker

def make_process_list():
    """
    전체 영상 목록과 처리할 영상 목록 생성
    """
    log_message("Creating process lists...")

    all_videos = []
    for root, _, files in os.walk(RAW_DIR):
        for f in sorted(files):
            if f.lower().endswith(".mp4"):
                rel = os.path.relpath(os.path.join(root, f), RAW_DIR)
                all_videos.append(rel)

    # video_list.txt 새로 생성
    os.makedirs(os.path.dirname(VIDEO_LIST_PATH), exist_ok=True)
    with open(VIDEO_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(all_videos))
    log_message(f"video_list.txt regenerated with {len(all_videos)} videos")

    # processed.txt 로드
    processed = load_set_from_file(PROCESSED_LIST_PATH)

    # 처리하지 않은 영상만 to_process.txt 생성
    to_process = [v for v in all_videos if v not in processed]
    os.makedirs(os.path.dirname(TOPROCESS_LIST_PATH), exist_ok=True)
    with open(TOPROCESS_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(to_process))
    log_message(f"to_process.txt generated with {len(to_process)} videos")

def extract_landmarks(max_count=None):
    """to_process.txt 기반 랜드마크 추출"""
    tracker = HandTracker()

    with open(TOPROCESS_LIST_PATH, "r", encoding="utf-8") as f:
        work_list = [line.strip() for line in f.readlines() if line.strip()]

    start = perf_counter()
    count = 0
    total = len(work_list)
    log_message(f"Start extracting landmarks for {total} videos...")

    for idx, rel_path in enumerate(work_list):
        log_message(f"[{idx+1}/{total}] Processing {rel_path}")
        video_path = os.path.join(RAW_DIR, rel_path)

        subfolder, filename = os.path.split(rel_path)
        video_id = os.path.splitext(filename)[0]

        landmark_save_dir = os.path.join(LANDMARKS_DIR, subfolder, video_id)
        draw_save_dir = os.path.join(DRAW_IMG_DIR, subfolder, video_id)
        os.makedirs(landmark_save_dir, exist_ok=True)
        os.makedirs(draw_save_dir, exist_ok=True)

        shutil.rmtree(FRAME_DIR, ignore_errors=True)
        os.makedirs(FRAME_DIR, exist_ok=True)

        try:
            frame_count = extract_frames_from_video(video_path, FRAME_DIR)
            frame_files = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg"))

            for frame_file in frame_files:
                frame_path = os.path.join(FRAME_DIR, frame_file)
                landmarks = tracker.process_image(frame_path)

                # npy로 저장
                save_path = os.path.join(
                    landmark_save_dir,
                    os.path.splitext(frame_file)[0] + ".npy"
                )
                tracker.save_landmarks(landmarks, save_path)

                # 시각화 저장
                draw_path = os.path.join(draw_save_dir, frame_file)
                tracker.draw_and_save_landmarks(frame_path, draw_path, landmarks)

            # 성공했으므로 extracted.txt에 추가
            append_line_to_file(EXTRACTED_LIST_PATH, rel_path)

        except Exception as e:
            log_message(f"[ERROR] {rel_path} 처리 실패: {e}")
            append_line_to_file(FAILED_VIDEOS_PATH, rel_path)

        # 실패 여부 상관없이 processed.txt에 기록
        append_line_to_file(PROCESSED_LIST_PATH, rel_path)
        count += 1
        if max_count and count >= max_count:
            break

    tracker.close()
    elapsed = perf_counter() - start
    log_message(f"Extraction finished. Processed {count}/{total} videos in {elapsed:.2f}s")


if __name__ == "__main__":
    make_process_list()
    extract_landmarks(max_count=None)
