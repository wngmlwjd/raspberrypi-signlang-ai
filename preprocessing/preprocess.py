import os
import glob
import json
from time import perf_counter
from collections import Counter
import random
import math
from typing import Tuple, List, Dict
import numpy as np

# utils 및 config 파일 경로는 그대로 유지 (외부 파일로 가정)
from utils import log_message
from preprocessing.config import (
    USE_LABELS_LIST_PATH, FPS, RAW_DIR,
    ALL_MORPHEME_DIRS, SEQUENCE_LENGTH, SEQUENCE_STEP,
    TRAIN_TEST_SPLIT, LANDMARKS_DIR,
    TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR,
    TEST_FEATURES_DIR, TEST_LABELS_DIR
)

# ----------------------------
# 0. WORD 번호와 REAL 번호에 따른 NPY 폴더 결정 함수
# ----------------------------
def get_npy_folder_from_metadata(word_num: int, real_num_str: str) -> str:
    """WORD 번호 범위와 REAL 번호에 따라 NPY 파일이 저장된 폴더 번호를 반환합니다."""
    
    # 0~1500이면 REAL 번호 뒤에 -1이 붙음
    if 0 <= word_num <= 1500:
        return f"{real_num_str}-1"
    
    # 1501 이상이면 REAL 번호만 사용
    elif word_num > 1500:
        return real_num_str
    
    # 그 외의 경우 (음수 등)
    else:
        return "99_UNKNOWN_RANGE" 

# ----------------------------
# 1. 라벨 매핑
# ----------------------------
def load_label_mapping():
    """사용할 라벨 목록을 불러오고 정수형 맵핑을 생성합니다."""
    with open(USE_LABELS_LIST_PATH, "r", encoding="utf-8") as f:
        use_labels_list = [line.strip() for line in f if line.strip()]
    label_to_int = {label: i for i, label in enumerate(use_labels_list)}
    log_message(f"✅ Loaded {len(use_labels_list)} labels from {os.path.basename(USE_LABELS_LIST_PATH)}")
    return label_to_int

# ----------------------------
# 2. JSON 스캔 및 데이터 수집 (NPY 경로 최종 수정 적용)
# ----------------------------
def scan_and_filter_data(label_to_int: dict):
    """라벨 JSON 파일을 스캔하고, 유효한 수어 구간을 추출하며, 데이터를 균형화합니다."""
    max_morpheme_frames = 0
    all_label_data = []

    total_morpheme_dirs = len(ALL_MORPHEME_DIRS)
    log_message(f"🔍 Starting scan across {total_morpheme_dirs} morpheme directories...")
    
    # 디버깅: NPY 파일 누락 라벨 기록용
    missing_npy_files_counter = Counter()

    for i, morpheme_dir_with_speaker in enumerate(ALL_MORPHEME_DIRS):
        # 진행 상황 출력
        if (i + 1) % 50 == 0 or (i + 1) == total_morpheme_dirs:
            log_message(f"   ... Scanning progress: {i + 1}/{total_morpheme_dirs} directories processed.")
            
        label_files = glob.glob(os.path.join(morpheme_dir_with_speaker, "**", "*.json"), recursive=True)
        for label_path in label_files:
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    full_json_data = json.load(f)
            except Exception as e:
                log_message(f"[WARN] Failed to load JSON {label_path}: {e}")
                continue

            sign_segments = full_json_data.get("data", [])
            if not sign_segments:
                continue

            for segment in sign_segments:
                if not segment.get("attributes"):
                    continue
                word = segment["attributes"][0].get("name")
                
                # 1. 라벨 불일치 체크
                if not word or word not in label_to_int:
                    continue

                start_frame = int(segment.get("start", 0) * FPS)
                end_frame = int(segment.get("end", 0) * FPS)
                morpheme_frames = end_frame - start_frame + 1
                if morpheme_frames <= 0:
                    continue

                max_morpheme_frames = max(max_morpheme_frames, morpheme_frames)

                # npy 파일 경로 매핑
                json_name = os.path.basename(label_path)
                base_name = json_name.replace("_morpheme.json", "")
                
                # ----------------------------------------------------
                # ⭐ NPY 파일 경로 매핑 로직 ⭐
                # ----------------------------------------------------
                # 1. WORD 번호 추출
                try:
                    word_part = base_name.split("_")[2] 
                    word_num = int(word_part.replace("WORD", ""))
                except (IndexError, ValueError):
                    log_message(f"[WARN] Failed to parse WORD num from {base_name}")
                    continue
                
                # 2. REAL 번호 추출
                try:
                    real_num_str = base_name.split("_REAL")[1].split("_")[0]
                except (IndexError, ValueError):
                    log_message(f"[WARN] Failed to parse REAL num from {base_name}")
                    continue

                # 3. WORD 번호와 REAL 번호에 따라 NPY 폴더 결정
                npy_root_folder = get_npy_folder_from_metadata(word_num, real_num_str)
                
                # 4. NPY 디렉토리 경로 구성
                npy_dir = os.path.join(LANDMARKS_DIR, npy_root_folder, base_name)
                    
                # ----------------------------------------------------

                # 5. NPY 파일 존재 여부 체크
                npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
                
                if not npy_files:
                    # ⭐ 디버깅 로직: NPY 파일이 없는 경우 경로 출력 ⭐
                    missing_npy_files_counter[word] += 1
                    log_message(f"[DEBUG_NPY_MISSING] Word: {word}, Expected NPY Dir: {npy_dir}, JSON Path: {label_path}")
                    continue

                all_label_data.append({
                    "word": word,
                    "int_label": label_to_int[word],
                    "start_frame": start_frame,
                    "end_frame": min(end_frame, len(npy_files)-1),
                    "npy_files": npy_files
                })

    # 언더샘플링 & 밸런싱
    data_by_label = {}
    for item in all_label_data:
        word = item['word']
        data_by_label.setdefault(word, []).append(item)

    if not data_by_label:
        log_message("❌ No valid data found for any label.")
        return 0, [], {} # max_len, balanced_data, data_by_label 반환 구조 변경 고려
    
    # 스캔 후 원본 데이터 개수 정보 반환
    original_data_counts = {word: len(items) for word, items in data_by_label.items()}

    # 디버깅: NPY 파일 누락 요약 출력
    if missing_npy_files_counter:
        log_message("--- Summary of Missing NPY Files by Word ---")
        for word, count in missing_npy_files_counter.most_common():
            # 최종 데이터에 남지 않은 라벨만 출력하여 문제 라벨에 집중
            if word not in data_by_label:
                 log_message(f"  > {word}: {count} occurrences missing NPY files.")
        log_message("--------------------------------------------")
        
    min_count = min(len(v) for v in data_by_label.values())
    balanced_data = []
    
    log_message(f"📊 Original total samples: {len(all_label_data)}")
    log_message(f"⚖️ Balancing data. Minimum samples per label: {min_count}")
    
    for word, items in data_by_label.items():
        random.shuffle(items)
        balanced_data.extend(items[:min_count])
        
    log_message(f"✅ Scanning and balancing finished. Final balanced samples: {len(balanced_data)}")
    return max_morpheme_frames, balanced_data

# ----------------------------
# 3. 프레임 정규화
# ----------------------------
def normalize_frames(data_list: list, max_frames: int):
    """모든 시퀀스의 길이를 가장 긴 길이(max_frames)에 맞게 프레임을 확장합니다."""
    log_message(f"📐 Normalizing sequence lengths to {max_frames} frames.")
    for item in data_list:
        start, end = item['start_frame'], item['end_frame']
        current_length = end - start + 1
        
        # 여기서 '시작/끝을 동일하게 늘리기' 전략 대신,
        # 최대 길이로 맞추기 위한 padding 계산만 수행 (보간 없이)
        padding_needed = max_frames - current_length
        pad_start = math.floor(padding_needed / 2)
        
        # 0 프레임보다 작아지지 않도록 조정
        new_start = max(start - pad_start, 0)
        # new_end는 L_max 길이만큼 보장하도록 계산
        new_end = new_start + max_frames - 1
        
        # 원본 npy_files 길이를 넘지 않도록 조정
        max_valid_frame = len(item['npy_files']) - 1
        if new_end > max_valid_frame:
             new_end = max_valid_frame
             new_start = new_end - max_frames + 1
             new_start = max(new_start, 0) # 다시 0보다 작아지지 않도록 확인

        item['normalized_start_frame'] = new_start
        item['normalized_end_frame'] = new_end
        
    log_message("✅ Normalization complete.")
    return data_list

# ----------------------------
# 4. 슬라이딩 윈도우
# ----------------------------
def create_sliding_windows(data_list: List[Dict], sequence_length: int, sequence_step: int) -> List[Dict]:
    """정규화된 시퀀스를 슬라이딩 윈도우 방식으로 분할합니다."""
    sequences = []
    seq_id = 0
    for item in data_list:
        norm_start, norm_end = item['normalized_start_frame'], item['normalized_end_frame']
        # 시작 프레임 계산 (end_frame - sequence_length + 1 까지 포함)
        for start_frame in range(norm_start, norm_end - sequence_length + 2, sequence_step): 
            end_frame = start_frame + sequence_length - 1
            
            # 윈도우가 정규화된 시퀀스 범위 안에 완전히 들어오는지 확인
            if end_frame <= norm_end: 
                seq_item = item.copy()
                seq_item.update({
                    "sequence_id": seq_id,
                    "sequence_start_frame": start_frame,
                    "sequence_end_frame": end_frame,
                    "sequence_length": sequence_length
                })
                sequences.append(seq_item)
                seq_id += 1
    return sequences

# ----------------------------
# 5. 좌표 변환 및 정규화
# ----------------------------
def transform_and_normalize_landmarks(landmarks_array: np.ndarray) -> np.ndarray:
    """프레임별 랜드마크를 손목 기준으로 변환하고 정규화합니다."""
    if landmarks_array.size == 0:
        return landmarks_array
        
    # 손목(0번 인덱스)을 기준점(0, 0, 0)으로 설정하여 상대 좌표로 변환
    reference_point = landmarks_array[:, 0:1, :] # (T, 1, 3)
    transformed = landmarks_array - reference_point
    
    # 손 크기를 기준으로 정규화 (최대 유클리드 거리로 나눔)
    max_norm = np.max(np.linalg.norm(transformed, axis=-1, keepdims=True))
    
    if max_norm < 1e-6:
        # 0으로 나누는 것을 방지
        return transformed
        
    return transformed / max_norm

# ----------------------------
# 6. 시퀀스 저장 (진행 상황 추가)
# ----------------------------
def save_sequences(sequences: List[Dict], features_dir: str, labels_dir: str, name: str):
    """슬라이딩 윈도우 시퀀스를 .npy 파일로 저장합니다."""
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    total_sequences = len(sequences)
    log_message(f"💾 Starting to save {name} sequences ({total_sequences} total) to {os.path.basename(features_dir)}/")

    for i, seq_item in enumerate(sequences):
        # 진행 상황 출력
        if (i + 1) % 1000 == 0 or (i + 1) == total_sequences:
            log_message(f"   ... Saving {name} sequences: {i + 1}/{total_sequences} processed.")
            
        npy_files = seq_item['npy_files']
        start, end = seq_item['sequence_start_frame'], seq_item['sequence_end_frame']
        seq_frames = []

        # 시퀀스 길이만큼 npy 파일 로드 및 전처리
        for f in npy_files[start:end+1]:
            frame = np.load(f)

            # 빈 배열 처리 (패딩용)
            if frame.size == 0:
                frame = np.zeros((1, 21, 3), dtype=np.float32)

            # 차원 맞춤: (J, 3) -> (1, J, 3)
            if frame.ndim == 2:
                frame = frame[np.newaxis, :, :] 
            elif frame.ndim == 1:
                # 1차원 데이터가 들어올 경우 (매우 드물지만 안전장치)
                frame = np.zeros((1, 21, 3), dtype=np.float32)

            # 손 랜드마크 21개 맞춤 (패딩)
            if frame.shape[1] < 21:
                pad = np.zeros((frame.shape[0], 21 - frame.shape[1], frame.shape[2]), dtype=frame.dtype)
                frame = np.concatenate([frame, pad], axis=1)

            seq_frames.append(frame)

        # 시퀀스 결합 및 최종 정규화
        seq_array = np.vstack(seq_frames)  # (T, J, 3)
        seq_array = transform_and_normalize_landmarks(seq_array)
        
        # LSTM 입력에 맞게 2차원으로 펼침 (T, J*C) -> (SEQUENCE_LENGTH, 63)
        seq_flat = seq_array.reshape(seq_array.shape[0], -1) 

        feat_path = os.path.join(features_dir, f"{seq_item['word']}_{seq_item['sequence_id']}.npy")
        lbl_path = os.path.join(labels_dir, f"{seq_item['word']}_{seq_item['sequence_id']}.txt")

        # 파일 저장
        np.save(feat_path, seq_flat)
        with open(lbl_path, "w") as f:
            f.write(str(seq_item['int_label']))
            
    log_message(f"✅ Saving {name} sequences complete.")

# ----------------------------
# 7. Train/Test split
# ----------------------------
def split_data_by_label(data_list: List[Dict], split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """라벨별로 훈련/테스트 셋을 분리합니다."""
    data_by_label = {}
    for item in data_list:
        word = item['word']
        data_by_label.setdefault(word, []).append(item)

    train_data, test_data = [], []
    random.seed(42) # 재현성을 위해 시드 고정
    
    log_message(f"✂️ Splitting data by label with ratio: {split_ratio:.2f} (Train) / {1-split_ratio:.2f} (Test)")

    for word, items in data_by_label.items():
        random.shuffle(items)
        split_point = int(len(items) * split_ratio)
        
        train_data.extend(items[:split_point])
        test_data.extend(items[split_point:])
        
        log_message(f"   ... {word}: Train={len(items[:split_point])}, Test={len(items[split_point:])}")

    log_message(f"✅ Split complete. Total Train meta: {len(train_data)}, Total Test meta: {len(test_data)}")
    return train_data, test_data

# ----------------------------
# 9. 시퀀스 길이 확인 및 로그 출력
# ----------------------------
def check_sequence_lengths(sequences: List[Dict], name: str):
    """모든 시퀀스의 길이가 동일한지 확인하고, 다른 길이가 있는 경우 출력."""
    lengths = [seq['sequence_end_frame'] - seq['sequence_start_frame'] + 1 for seq in sequences]
    unique_lengths = set(lengths)
    
    if len(unique_lengths) == 1:
        log_message(f"✅ All {name} sequences have consistent length: {unique_lengths.pop()} frames")
    else:
        log_message(f"[WARN] {name} sequences have inconsistent lengths: {sorted(unique_lengths)}")
        # 길이별 몇 개씩 있는지도 출력
        length_counts = Counter(lengths)
        for length, count in sorted(length_counts.items()):
            log_message(f"  > Length {length}: {count} sequences")

# ----------------------------
# 8. 실행
# ----------------------------
if __name__ == "__main__":
    start_time = perf_counter()
    log_message("--- Start Data Pipeline ---")

    label_map = load_label_mapping()
    
    # ----------------------------------------------------
    # 라벨 매핑 및 초기화
    # ----------------------------------------------------
    log_message("--- Loaded Label Mapping ---")
    if label_map:
        sorted_labels = sorted(label_map.items(), key=lambda item: item[1])
        for label, index in sorted_labels:
            log_message(f"  > Index {index:2d}: {label}")
    log_message("----------------------------")
    
    # ----------------------------------------------------
    # 데이터 스캔 및 처리
    # ----------------------------------------------------
    if label_map:
        max_len, balanced_data = scan_and_filter_data(label_map)
        
        if max_len > 0 and balanced_data:
            # 3. 프레임 정규화
            final_data = normalize_frames(balanced_data, max_len)

            # 7. Train/Test split
            train_data_meta, test_data_meta = split_data_by_label(final_data, split_ratio=TRAIN_TEST_SPLIT)
            
            # 4. 슬라이딩 윈도우 생성
            log_message(f"📏 Creating sequences (Length={SEQUENCE_LENGTH}, Step={SEQUENCE_STEP})...")
            train_sequences = create_sliding_windows(train_data_meta, SEQUENCE_LENGTH, SEQUENCE_STEP)
            test_sequences = create_sliding_windows(test_data_meta, SEQUENCE_LENGTH, SEQUENCE_STEP)
            log_message(f"✅ Sequence creation complete. Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")

            # ----------------------------------------------------
            # ⭐ 라벨별 최종 시퀀스 개수 출력 (추가된 부분) ⭐
            # ----------------------------------------------------
            train_counts = Counter(item['word'] for item in train_sequences)
            test_counts = Counter(item['word'] for item in test_sequences)
            
            log_message("\n--- Final Sequence Counts by Label ---")
            
            # 라벨 맵핑 순서대로 출력
            for label, index in sorted_labels:
                train_count = train_counts.get(label, 0)
                test_count = test_counts.get(label, 0)
                log_message(f"  > {label} (Idx {index:2d}): Train={train_count:,} sequences, Test={test_count:,} sequences")
            
            total_train_seq = sum(train_counts.values())
            total_test_seq = sum(test_counts.values())
            log_message(f"--- TOTAL: Train={total_train_seq:,} sequences, Test={total_test_seq:,} sequences ---")
            
            # ----------------------------------------------------
            
            # 6. 시퀀스 저장
            save_sequences(train_sequences, TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR, name="TRAIN")
            save_sequences(test_sequences, TEST_FEATURES_DIR, TEST_LABELS_DIR, name="TEST")

            log_message(f"✅ Final save complete. Total train sequences: {len(train_sequences)}, Total test sequences: {len(test_sequences)}.")
        else:
            log_message("[WARN] No valid data found after scanning and balancing.")
            
    log_message(f"--- Pipeline finished in {perf_counter() - start_time:.2f}s ---")
    
    check_sequence_lengths(train_sequences, "TRAIN")
    check_sequence_lengths(test_sequences, "TEST")