"""
랜드마크 재검증 및 필요 시 재추출/정리 스크립트
- npy 파일이 초과된 경우 삭제
- npy 파일이 부족한 경우 재추출
- 프레임 수와 npy 파일 수 로그 출력
- 전처리 완료 후 원본 영상 삭제
- 총 영상 수 및 처리한 영상 수 출력
"""

import os
import shutil
from time import perf_counter

from utils import log_message, append_line_to_file, load_set_from_file
from landmark_extractor.config import (
    RAW_DIR,
    FRAME_DIR_1,
    LANDMARKS_DIR,
    PROCESSED_LIST_PATH,
    EXTRACTED_LIST_PATH,
    FAILED_VIDEOS_PATH,
)
from landmark_extractor.utils import extract_frames_from_video
from landmark_extractor.hand_tracking import HandTracker


def get_landmark_dir_from_rel(rel_path):
    parts = rel_path.split(os.sep)
    video_id = os.path.splitext(parts[-1])[0]
    if len(parts) >= 3:
        return os.path.join(*parts[1:-1], video_id)
    else:
        return video_id


def count_saved_landmarks(landmark_dir):
    if not os.path.isdir(landmark_dir):
        return 0
    return len([f for f in os.listdir(landmark_dir) if f.endswith(".npy")])


def delete_excess_npys(landmark_dir, expected_count):
    """프레임 수보다 많은 npy 파일 삭제"""
    npy_files = sorted([f for f in os.listdir(landmark_dir) if f.endswith(".npy")])
    excess_count = len(npy_files) - expected_count
    if excess_count > 0:
        for f in npy_files[expected_count:]:
            os.remove(os.path.join(landmark_dir, f))
        log_message(f"[CLEANED] Deleted {excess_count} excess npy files in {landmark_dir}")


def reprocess_video(rel_path, tracker):
    video_path = os.path.join(RAW_DIR, rel_path)
    landmark_subpath = get_landmark_dir_from_rel(rel_path)
    landmark_save_dir = os.path.join(LANDMARKS_DIR, landmark_subpath)

    shutil.rmtree(landmark_save_dir, ignore_errors=True)
    os.makedirs(landmark_save_dir, exist_ok=True)

    shutil.rmtree(FRAME_DIR_1, ignore_errors=True)
    os.makedirs(FRAME_DIR_1, exist_ok=True)

    try:
        frame_count = extract_frames_from_video(video_path, FRAME_DIR_1)
        frame_files = sorted([f for f in os.listdir(FRAME_DIR_1) if f.endswith(".jpg")])

        for frame_file in frame_files:
            frame_path = os.path.join(FRAME_DIR_1, frame_file)
            landmarks = tracker.process_image(frame_path)
            save_path = os.path.join(
                landmark_save_dir,
                os.path.splitext(frame_file)[0] + ".npy"
            )
            tracker.save_landmarks(landmarks, save_path)

        append_line_to_file(EXTRACTED_LIST_PATH, rel_path)
        log_message(f"[RE-EXTRACTED] {rel_path}")
        return True

    except Exception as e:
        log_message(f"[ERROR][REPROCESS] {rel_path}: {e}")
        append_line_to_file(FAILED_VIDEOS_PATH, rel_path)
        return False


def recheck_and_clean(delete_original=False):
    log_message("===== Start Recheck & Clean =====")

    processed = load_set_from_file(PROCESSED_LIST_PATH)
    extracted = load_set_from_file(EXTRACTED_LIST_PATH)
    failed = load_set_from_file(FAILED_VIDEOS_PATH)

    total_videos = len(processed)
    processed_count = 0

    tracker = HandTracker()

    for idx, rel_path in enumerate(sorted(processed), start=1):
        video_path = os.path.join(RAW_DIR, rel_path)
        landmark_subpath = get_landmark_dir_from_rel(rel_path)
        landmark_dir = os.path.join(LANDMARKS_DIR, landmark_subpath)

        # 🔥 원본 영상 없는 경우는 조용히 skip
        if not os.path.isfile(video_path):
            continue

        # 🔥 진행률 출력
        log_message(f"[PROGRESS] {idx} / {total_videos} processing: {rel_path}")

        try:
            expected_frames = extract_frames_from_video(video_path, FRAME_DIR_1)
        except Exception as e:
            log_message(f"[ERROR] Cannot read video: {rel_path} → {e}")
            failed.add(rel_path)
            extracted.discard(rel_path)
            continue

        saved_npys = count_saved_landmarks(landmark_dir)
        log_message(f"[CHECK] {rel_path}: frames={expected_frames}, npys={saved_npys}")

        changed = False

        if saved_npys > expected_frames:
            delete_excess_npys(landmark_dir, expected_frames)
            saved_npys = count_saved_landmarks(landmark_dir)
            log_message(f"[AFTER CLEAN] {rel_path}: npys={saved_npys}")
            changed = True

        if saved_npys < expected_frames:
            log_message(f"[REPROCESS] {rel_path} - npys missing ({saved_npys}/{expected_frames})")
            extracted.discard(rel_path)
            failed.discard(rel_path)
            ok = reprocess_video(rel_path, tracker)
            if ok:
                extracted.add(rel_path)
                changed = True
            else:
                failed.add(rel_path)

        if delete_original and saved_npys == expected_frames:
            try:
                os.remove(video_path)
                log_message(f"[DELETED] Original video: {video_path}")
                changed = True
            except Exception as e:
                log_message(f"[ERROR] Cannot delete video: {video_path} → {e}")

        if changed:
            processed_count += 1

    tracker.close()

    with open(EXTRACTED_LIST_PATH, "w", encoding="utf-8") as f:
        for item in sorted(extracted):
            f.write(item + "\n")

    with open(FAILED_VIDEOS_PATH, "w", encoding="utf-8") as f:
        for item in sorted(failed):
            f.write(item + "\n")

    log_message(f"===== Recheck & Clean Completed =====")
    log_message(f"Total videos listed: {total_videos}")
    log_message(f"Videos processed (cleaned/re-extracted/deleted): {processed_count}")



if __name__ == "__main__":
    # delete_original=True로 설정하면 프레임과 npy 정합성 확인 후 원본 영상 삭제
    recheck_and_clean(delete_original=True)
