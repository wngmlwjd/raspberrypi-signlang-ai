import os
import glob
import json
import math
import random
import numpy as np
from collections import Counter
from time import perf_counter
from typing import List, Dict, Tuple
import shutil
import re

from preprocess.config import (
    USE_WORD_NUM, LABELS_LIST_PATH, FPS, LANDMARKS_DIR,
    TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR,
    TEST_FEATURES_DIR, TEST_LABELS_DIR,
    ALL_MORPHEME_DIRS, SEQUENCE_LENGTH, SEQUENCE_STEP,
    TRAINING_VIDEOS_LIST, VALIDATION_VIDEOS_LIST
)
from utils import log_message

# ----------------------------
# 1. 라벨 로드 및 integer 매핑
# ----------------------------
def load_label_mapping():
    with open(LABELS_LIST_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        labels = [line.strip().split()[0] for line in lines if line.strip()]
    labels = labels[:USE_WORD_NUM]
    label_to_int = {label: idx for idx, label in enumerate(labels)}
    print("✅ Selected labels:", ", ".join(labels))
    log_message(f"✅ Loaded {len(labels)} labels (USE_WORD_NUM={USE_WORD_NUM}) from {LABELS_LIST_PATH}")
    return labels, label_to_int

# ----------------------------
# 2. 좌표 변환 및 정규화
# ----------------------------
def transform_and_normalize_landmarks(landmarks_array: np.ndarray) -> np.ndarray:
    if landmarks_array.size == 0:
        return landmarks_array
    ref = landmarks_array[:, 0:1, :]
    transformed = landmarks_array - ref
    max_norm = np.max(np.linalg.norm(transformed, axis=-1, keepdims=True))
    if max_norm < 1e-6:
        return transformed
    return transformed / max_norm

# ----------------------------
# 3. npy 폴더 결정
# ----------------------------
def get_npy_dir_from_json(json_filename: str) -> str:
    base_name = os.path.basename(json_filename).replace("_morpheme.json", "")
    real_match = re.search(r'_REAL(\d+)', base_name)
    if not real_match:
        raise ValueError(f"Cannot find REAL number in {base_name}")
    speaker_num = real_match.group(1)
    word_match = re.search(r'_WORD(\d+)', base_name)
    if not word_match:
        raise ValueError(f"Cannot find WORD number in {base_name}")
    word_num = int(word_match.group(1))
    if 1 <= word_num <= 1499:
        speaker_folder = f"{speaker_num}-1"
    else:
        speaker_folder = speaker_num
    npy_dir = os.path.join(LANDMARKS_DIR, speaker_folder, base_name)
    return npy_dir

# ----------------------------
# 4. JSON 스캔 및 max frame 추출
# ----------------------------
def scan_all_jsons(label_to_int: dict) -> Tuple[int, List[Dict]]:
    all_data = []
    max_frames = 0
    for morpheme_dir in ALL_MORPHEME_DIRS:
        json_files = glob.glob(os.path.join(morpheme_dir, "**", "*.json"), recursive=True)
        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue
            segments = data.get("data", [])
            if not segments:
                continue
            for seg in segments:
                attrs = seg.get("attributes")
                if not attrs:
                    continue
                word = attrs[0].get("name")
                if not word or word not in label_to_int:
                    continue
                start_time_sec = seg.get("start", 0)
                end_time_sec = seg.get("end", 0)
                start_frame = int(start_time_sec * FPS)
                end_frame = int(end_time_sec * FPS)
                frame_count = end_frame - start_frame + 1
                if frame_count <= 0:
                    continue
                max_frames = max(max_frames, frame_count)
                try:
                    npy_dir = get_npy_dir_from_json(json_path)
                except ValueError as e:
                    log_message(f"⚠️ {e}")
                    continue
                npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
                if not npy_files:
                    log_message(f"⚠️ npy files not found: {npy_dir}")
                    continue
                all_data.append({
                    "word": word,
                    "int_label": label_to_int[word],
                    "npy_files": npy_files,
                    "start_frame": start_frame,
                    "end_frame": min(end_frame, len(npy_files)-1)
                })
    log_message(f"✅ Scan complete. Max frames across all segments: {max_frames}")
    return max_frames, all_data

# ----------------------------
# 5. 시퀀스 정규화
# ----------------------------
def normalize_sequences(data_list: List[Dict], max_frames: int) -> List[Dict]:
    for item in data_list:
        cur_len = item['end_frame'] - item['start_frame'] + 1
        pad = max_frames - cur_len
        pad_start = pad // 2
        new_start = max(item['start_frame'] - pad_start, 0)
        new_end = new_start + max_frames - 1
        new_end = min(new_end, len(item['npy_files'])-1)
        new_start = new_end - max_frames + 1
        item['normalized_start_frame'] = new_start
        item['normalized_end_frame'] = new_end
    return data_list

# ----------------------------
# 6. 슬라이딩 윈도우
# ----------------------------
def create_sliding_windows(data_list: List[Dict]) -> List[Dict]:
    sequences = []
    seq_id = 0
    for item in data_list:
        start, end = item['normalized_start_frame'], item['normalized_end_frame']
        seq_start = start
        while seq_start <= end:
            seq_end = seq_start + SEQUENCE_LENGTH - 1
            # 🔹 마지막 시퀀스가 end를 넘어가면, 부족한 길이만큼 뒤쪽 패딩
            if seq_end > end:
                seq_end = end
            seq_item = item.copy()
            seq_item.update({
                "sequence_id": seq_id,
                "sequence_start_frame": seq_start,
                "sequence_end_frame": seq_end,
                "sequence_length": seq_end - seq_start + 1  # 실제 프레임 길이
            })
            sequences.append(seq_item)
            seq_id += 1
            seq_start += SEQUENCE_STEP
    log_message(f"✅ Created {len(sequences)} sequences.")
    return sequences

def save_sequences(sequences: List[Dict], feat_dir: str, label_dir: str) -> int:
    if os.path.exists(feat_dir):
        shutil.rmtree(feat_dir)
    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    saved_files = []
    for item in sequences:
        seq_files = item['npy_files'][item['sequence_start_frame']:item['sequence_end_frame']+1]
        seq_arrays = []

        # 🔹 먼저 시퀀스 내 각 프레임의 J 확인
        Js = []
        for f in seq_files:
            try:
                arr = np.load(f)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                Js.append(arr.shape[1])
            except:
                continue
        J_max = max(Js) if Js else 63  # default 63

        # 🔹 시퀀스 프레임 처리 및 J_max로 맞춤
        for f in seq_files:
            try:
                arr = np.load(f)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                # J가 부족하면 뒤에 zero padding
                if arr.shape[1] < J_max:
                    pad_width = J_max - arr.shape[1]
                    arr = np.pad(arr, ((0,0),(0,pad_width),(0,0)), mode='constant')
                # J가 초과하면 잘라서 맞춤
                elif arr.shape[1] > J_max:
                    arr = arr[:, :J_max, :]
                arr = transform_and_normalize_landmarks(arr)
            except:
                arr = np.zeros((1, J_max, 3), dtype=np.float32)
            seq_arrays.append(arr)

        # 🔹 SEQUENCE_LENGTH 맞추기 (항상 SEQUENCE_LENGTH 프레임 확보)
        seq_arrays_padded = []
        for i in range(SEQUENCE_LENGTH):
            if i < len(seq_arrays):
                seq_arrays_padded.append(seq_arrays[i][0])  # shape: (J_max, 3)
            else:
                seq_arrays_padded.append(np.zeros((J_max, 3), dtype=np.float32))

        # 🔹 stack으로 (SEQUENCE_LENGTH, J_max, 3) 보장
        seq_array = np.stack(seq_arrays_padded, axis=0)

        # 🔹 최종 feature flatten: (SEQUENCE_LENGTH, J_max*3)
        seq_flat = seq_array.reshape(SEQUENCE_LENGTH, -1)

        base_fname = f"{item['word']}_{item['sequence_id']}"
        feat_path = os.path.join(feat_dir, base_fname + ".npy")
        label_path = os.path.join(label_dir, base_fname + ".txt")

        np.save(feat_path, seq_flat)
        with open(label_path, "w") as f:
            f.write(str(item['int_label']))

        saved_files.append((feat_path, label_path))

    log_message(f"✅ Saved {len(saved_files)} sequences.")
    return len(saved_files)

# ----------------------------
# 8. 화자 기준 train/test split
# ----------------------------
def split_train_test_by_speaker(data_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    train, test = [], []
    for item in data_list:
        npy_path = item['npy_files'][0]
        match = re.search(r'/(\d+)[/-]', npy_path)
        if not match:
            log_message(f"⚠️ Cannot extract speaker number from {npy_path}, skipping")
            continue
        speaker_num = int(match.group(1))
        if speaker_num in TRAINING_VIDEOS_LIST:
            train.append(item)
        elif speaker_num in VALIDATION_VIDEOS_LIST:
            test.append(item)
        else:
            log_message(f"⚠️ Speaker {speaker_num} not in train/test list, skipping")
    return train, test

# ----------------------------
# 9. 실행
# ----------------------------
if __name__ == "__main__":
    start_time = perf_counter()
    log_message("--- Start preprocessing pipeline ---")

    labels, label_map = load_label_mapping()
    max_frames, data = scan_all_jsons(label_map)
    normalized_data = normalize_sequences(data, max_frames)
    train_data, test_data = split_train_test_by_speaker(normalized_data)
    train_sequences = create_sliding_windows(train_data)
    test_sequences = create_sliding_windows(test_data)

    train_saved_count = save_sequences(train_sequences, TRAIN_FEATURES_DIR, TRAIN_LABELS_DIR)
    test_saved_count = save_sequences(test_sequences, TEST_FEATURES_DIR, TEST_LABELS_DIR)

    log_message(f"✅ Pipeline finished in {perf_counter() - start_time:.2f}s")
    log_message(f"Total train sequences saved: {train_saved_count}, Total test sequences saved: {test_saved_count}")
