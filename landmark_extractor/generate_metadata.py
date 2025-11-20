# generate_metadata.py
import os

# mediapipe 폴더 안 config와 utils 가져오기
from landmark_extractor.config import (
    RAW_DIR,
    LANDMARKS_DIR,
    METADATA_DIR,
    VIDEO_LIST_PATH,
    TOPROCESS_LIST_PATH,
    PROCESSED_LIST_PATH,
    EXTRACTED_LIST_PATH,
    FAILED_VIDEOS_PATH,
)
from landmark_extractor.utils import log_message

# -------------------------------
# 메타데이터 폴더 생성
# -------------------------------
os.makedirs(METADATA_DIR, exist_ok=True)

# -------------------------------
# 전체 영상 리스트 생성
# -------------------------------
video_list = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith((".mp4", ".avi", ".mov")):
            rel = os.path.relpath(os.path.join(root, f), RAW_DIR)
            video_list.append(rel)

video_list = sorted(video_list)

with open(VIDEO_LIST_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(video_list))
log_message(f"전체 영상 목록 생성: {len(video_list)}개")

# -------------------------------
# 이미 처리된 영상 확인 및 실패 영상 체크
# -------------------------------
processed = set()
extracted = set()
failed = set()

for rel in video_list:
    # 폴더 구조: LANDMARKS_DIR/<영상폴더>/video_id
    subfolder, filename = os.path.split(rel)
    video_id = os.path.splitext(filename)[0]

    # subfolder가 1.Training / 2.Validation 이면 제외
    # subfolder 이름 그대로 두면 안 되고, 그 바로 하위 폴더만 유지
    # rel이 "2.Validation/18/NIA_SL_WORD3000_REAL18_U.mp4" 라면
    # -> landmark_folder = LANDMARKS_DIR/18/NIA_SL_WORD3000_REAL18_U
    parts = rel.split(os.sep)
    if len(parts) >= 3:
        landmark_folder = os.path.join(LANDMARKS_DIR, parts[1], os.path.splitext(parts[-1])[0])
    else:
        landmark_folder = os.path.join(LANDMARKS_DIR, os.path.splitext(parts[-1])[0])

    # print(landmark_folder)
    if os.path.exists(landmark_folder):
        npy_files = [f for f in os.listdir(landmark_folder) if f.endswith(".npy")]
        processed.add(rel)
        if npy_files:
            extracted.add(rel)  # npy 파일이 있으면 처리 완료
        else:
            failed.add(rel)     # 폴더는 있지만 npy 파일이 없으면 실패 처리
    # 폴더가 없으면 처리되지 않은 상태로 두고 to_process에 포함됨

# -------------------------------
# 메타데이터 파일 저장
# -------------------------------
# processed.txt 저장
with open(PROCESSED_LIST_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(processed)))
log_message(f"이미 처리된 영상 목록 생성: {len(processed)}개")

# extracted.txt 저장
with open(EXTRACTED_LIST_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(extracted)))
log_message(f"추출된 영상 목록 생성: {len(extracted)}개")

# failed.txt 저장
with open(FAILED_VIDEOS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(failed)))
log_message(f"실패 영상 목록 생성: {len(failed)}개")

# 처리할 영상 목록 생성
to_process = [v for v in video_list if v not in extracted and v not in failed]
with open(TOPROCESS_LIST_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(to_process))
log_message(f"처리해야 하는 영상 목록 생성: {len(to_process)}개")

log_message("메타데이터 생성 완료")
