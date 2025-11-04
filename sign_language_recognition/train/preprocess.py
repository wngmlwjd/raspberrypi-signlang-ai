import numpy as np
import csv
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, Set, Dict, Any, List
from collections import defaultdict
import os 
import time # time 모듈 추가

# utils.py에서 정의된 경로, 상수 및 유틸리티 함수를 import
from sign_language_recognition.train.utils import ( 
    log_message, BASE_DIR, PROCESSED_DIR,
    TRAIN_MANIFEST_PATH, TRAIN_X_NPY_PATH, TRAIN_Y_NPY_PATH, ENCODER_PATH,
    VAL_MANIFEST_PATH, VAL_X_NPY_PATH, VAL_Y_NPY_PATH,
    PROCESSED_MANIFEST_TRAIN, PROCESSED_MANIFEST_VAL,
    SEQUENCE_LENGTH, SEQUENCE_STEP, MAX_HANDS, pad_landmark_array
)

def load_processed_manifest_keys(path: Path) -> Set[str]:
    """매니페스트 생성 단계에서 이미 처리 완료된 항목의 고유 키 (signer/video_name)를 로드합니다."""
    if not path.exists():
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        log_message(f"경고: 처리 완료 키 파일 로드 중 오류 발생 ({path.name}): {e}")
        return set()

def load_sequences_from_manifest(
    manifest_path: Path, 
    base_dir: Path, 
    seq_len: int, 
    seq_step: int,
    max_hands: int,
    # ✅ 증분 로딩을 위한 새로운 인자
    keys_to_process: Optional[Set[str]] = None 
) -> Tuple[np.ndarray, np.ndarray, Set[str]]:
    """
    manifest.csv 파일을 읽고, 'keys_to_process'에 해당하는 항목만 시퀀스 데이터를 로드하여 생성합니다.
    keys_to_process가 None이면 전체를 처리합니다.
    """
    sequences = []
    labels = []
    
    # ... (매니페스트 파일 존재 확인, 로드, 헤더 및 엔트리 개수 확인 로직 유지) ...
    if not manifest_path.exists():
        log_message(f"오류: 매니페스트 파일이 존재하지 않습니다: {manifest_path.absolute()}")
        return np.array([]), np.array([]), set()

    log_message(f"매니페스트 파일 로드 중: {manifest_path.name}")
    log_message(f"시퀀스 설정: 길이={seq_len} 프레임, 스텝(오버랩)={seq_step} 프레임") 
    
    manifest_entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        manifest_entries = list(reader)

    log_message(f"총 {len(manifest_entries)}개의 데이터 항목 발견.")
    unique_labels = set()
    
    entries_to_process_count = 0
    if keys_to_process is not None:
        entries_to_process_count = len(keys_to_process)
        log_message(f"✅ 증분 처리 모드: 처리할 새로운 항목 키 {entries_to_process_count}개.")
    
    
    for idx, entry in enumerate(manifest_entries):
        unique_key = f"{entry['signer_id']}/{entry['original_video_name']}"
        
        # ✅ 증분 로딩: 처리할 키 목록에 포함된 항목만 처리
        if keys_to_process is not None and unique_key not in keys_to_process:
            continue

        relative_path = Path(entry['landmark_folder_relative_path'])
        # 참고: base_dir은 './dataset'이므로, 전체 경로를 조합합니다.
        npy_data_dir = base_dir / relative_path 
        
        # ... (로그 및 로드 로직 유지) ...
        # 현재 처리 중인 항목이 증분 항목일 경우에만 진행 상황을 로그에 자주 출력합니다.
        if (keys_to_process is None) and ((idx + 1) % 100 == 0 or idx == 0 or idx == len(manifest_entries) - 1):
             log_message(f"[{idx+1}/{len(manifest_entries)}] 로드 중: 라벨='{entry['word_label']}', 경로={relative_path}")
        elif keys_to_process is not None:
            # 증분 모드에서는 처리되는 항목만 카운트하고 로그를 찍는 것이 더 효율적입니다.
            current_processed_idx = len(labels) 
            if current_processed_idx % 10 == 0:
                 log_message(f"[증분 {current_processed_idx}/{entries_to_process_count}] 로드 중: 라벨='{entry['word_label']}'")

        if not npy_data_dir.is_dir():
            log_message(f"경고: 랜드마크 폴더가 존재하지 않아 건너뜁니다: {npy_data_dir.name}")
            continue
            
        npy_files = sorted(list(npy_data_dir.glob('*.npy')))
        current_files = npy_files
        
        for i in range(0, len(current_files) - seq_len + 1, seq_step): 
            seq_frames = []
            files_slice = current_files[i:i + seq_len]

            for npy_file in files_slice:
                try:
                    lm = np.load(npy_file, allow_pickle=True)
                    padded_lm = pad_landmark_array(lm) 
                    seq_frames.append(padded_lm.flatten())
                except Exception as e:
                    log_message(f"경고: NPY 파일 로드 오류 ({npy_file.name}, 시퀀스 인덱스 {i}): {repr(e)}")
                    continue

            if len(seq_frames) == seq_len:
                sequences.append(np.array(seq_frames))
                labels.append(entry['word_label'])
                unique_labels.add(entry['word_label'])
                
    log_message(f"데이터 로드 완료. 총 시퀀스 수: {len(sequences)}, 총 라벨 수: {len(unique_labels)}")
    return np.array(sequences), np.array(labels), unique_labels


def _load_and_preprocess_single_dataset(
    data_type: str, # 'train' 또는 'val'
    manifest_path: Path,
    x_npy_path: Path,
    y_npy_path: Path,
    processed_manifest_keys_path: Path, # 매니페스트 생성 시 사용된 처리 완료 항목 기록
    label_encoder: Optional[LabelEncoder] = None,
    force_reprocess: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
    """
    단일 데이터셋 (훈련 또는 검증)을 로드하거나 manifest를 기반으로 생성/증분 업데이트합니다.
    """
    log_message(f"--- [{data_type.upper()}] 데이터셋 로딩 및 전처리 시작 ---")
    
    # Manifest의 모든 항목 키를 추출하는 헬퍼 함수
    def get_all_manifest_keys(m_path: Path) -> Set[str]:
        keys = set()
        if not m_path.exists():
            return keys
        with open(m_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for entry in reader:
                keys.add(f"{entry['signer_id']}/{entry['original_video_name']}")
        return keys

    X_old, Y_old = None, None
    is_incremental_mode = False
    
    # --------------------- 1. 기존 NPY 파일 로드 및 증분/재처리 여부 판단 ---------------------
    
    if x_npy_path.exists() and y_npy_path.exists() and not force_reprocess:
        try:
            # 1-1. 기존 데이터 로드 시도
            # mmap_mode='r'을 사용하여 메모리 사용량을 절약하고 파일에서 직접 읽습니다.
            X_old = np.load(x_npy_path, mmap_mode='r') 
            Y_old = np.load(y_npy_path, mmap_mode='r') 
            log_message(f"기존 NPY 데이터 로드 성공. Shape: X={X_old.shape}, Y={Y_old.shape}")
            
            # 1-2. 매니페스트 키 파일과 비교하여 증분 처리 필요성 판단
            
            # **A. NPY 파일이 최신인지 확인** (매니페스트 파일 변경 여부)
            manifest_time = os.path.getmtime(manifest_path) if manifest_path.exists() else 0
            npy_time = os.path.getmtime(x_npy_path)
            
            if manifest_time > npy_time:
                # 매니페스트가 NPY보다 최신인 경우: 증분 또는 전체 재처리 필요
                
                # **B. 매니페스트의 모든 키와 이전에 처리된 키 비교**
                all_keys_in_manifest = get_all_manifest_keys(manifest_path)
                processed_keys_in_txt = load_processed_manifest_keys(processed_manifest_keys_path)
                
                # NPY가 생성된 이후 매니페스트에 추가된 키 (새로운 비디오 항목)
                new_keys_to_process = all_keys_in_manifest - processed_keys_in_txt
                
                if not new_keys_to_process:
                    log_message("⚠️ 매니페스트는 최신이지만, NPY 생성 이후에 추가된 새 항목 키가 없어 증분 처리 건너뜀.")
                    # NPY 파일 수정 시간을 Manifest와 동기화 (다음번 실행 시 다시 검사하지 않도록)
                    os.utime(x_npy_path, (time.time(), time.path.getmtime(manifest_path))) 
                    return X_old, Y_old, label_encoder

                log_message(f"✅ 증분 처리 시작: 매니페스트에 {len(new_keys_to_process)}개의 새 항목이 추가됨.")
                is_incremental_mode = True
                
            else:
                log_message("NPY 파일이 최신이거나 매니페스트와 동기화되어 있습니다. 처리 건너뜀.")
                return X_old, Y_old, label_encoder
                
        except Exception as e:
            log_message(f"경고: NPY 파일 로드 또는 매니페스트 비교 중 오류 발생 ({repr(e)}). 전체 재처리합니다.")
            force_reprocess = True

    # --------------------- 2. 데이터 로드 및 처리 (재처리 또는 증분) ---------------------
    
    # 2-1. 재처리 모드 결정
    if force_reprocess:
        log_message("강제 재처리 모드 활성화. 전체 manifest를 기반으로 새로 생성합니다.")
        x_npy_path.unlink(missing_ok=True)
        y_npy_path.unlink(missing_ok=True)
        X_old, Y_old = None, None # np.array([]) 대신 None을 사용하여 명확히 구분
        keys_to_process = None # 전체 manifest 처리
        
    elif is_incremental_mode:
        log_message("증분 처리 모드 활성화. 새 항목만 로드합니다.")
        # 이전에 찾은 new_keys_to_process 사용
        keys_to_process = new_keys_to_process 
        
    else:
        # NPY 파일이 없는 경우, 전체 재처리
        log_message("사전 처리된 데이터셋이 없습니다. 전체 manifest를 기반으로 생성합니다.")
        keys_to_process = None


    # 2-2. raw 데이터 로드 (새 데이터 또는 전체 데이터)
    x_new_raw, y_new_raw_labels, _ = load_sequences_from_manifest(
        manifest_path=manifest_path, 
        base_dir=BASE_DIR, 
        seq_len=SEQUENCE_LENGTH, 
        seq_step=SEQUENCE_STEP,
        max_hands=MAX_HANDS,
        keys_to_process=keys_to_process # 증분 모드 시, 새 키만 전달
    )
    
    if x_new_raw.size == 0 and X_old is None:
        log_message("오류: 로드할 시퀀스 데이터가 없습니다. 처리를 중단합니다.")
        return None, None, label_encoder

    # 2-3. 라벨 인코딩 및 최종 병합
    X_final, Y_final = x_new_raw, None

    if X_old is not None:
        # 🟢 증분 모드
        if label_encoder is None:
             log_message("치명적 오류: 증분 모드인데 LabelEncoder가 로드되지 않았습니다. 처리를 중단합니다.")
             return None, None, label_encoder
             
        try:
            Y_new_encoded = label_encoder.transform(y_new_raw_labels)
        except ValueError as e:
            log_message(f"치명적 오류: 증분 데이터에 훈련 데이터에 없는 클래스가 포함되어 있습니다. {repr(e)}")
            return None, None, label_encoder
        
        # 데이터 병합 (기존 + 신규)
        X_final = np.concatenate((X_old, x_new_raw), axis=0)
        Y_final = np.concatenate((Y_old, Y_new_encoded), axis=0)
        
        log_message(f"증분 데이터 병합 완료. 최종 Shape: X={X_final.shape}, Y={Y_final.shape}")
        
        # X_old와 Y_old가 mmap_mode='r'로 로드되었을 수 있으므로, 메모리에 복사하여 저장합니다.
        X_final = X_final.copy()
        Y_final = Y_final.copy()

    else:
        # 🔴 전체 재처리 (force_reprocess=True 또는 NPY 파일이 없던 경우)
        if data_type == 'train':
            # 훈련 데이터인 경우에만 인코더를 새로 fit
            if label_encoder is None:
                label_encoder = LabelEncoder()
                Y_final = label_encoder.fit_transform(y_new_raw_labels) 
                log_message(f"[{data_type.upper()}] 새로운 LabelEncoder를 생성하고 fit했습니다. 클래스 수={len(label_encoder.classes_)}")
            else:
                # force_reprocess가 True이지만 외부에서 encoder를 전달받은 경우
                Y_final = label_encoder.transform(y_new_raw_labels)
                log_message(f"[{data_type.upper()}] 기존 LabelEncoder를 사용하여 transform했습니다.")
        elif data_type == 'val':
            # 검증 데이터인 경우 훈련 데이터에서 생성된 인코더를 사용
            if label_encoder is not None:
                try:
                    Y_final = label_encoder.transform(y_new_raw_labels)
                    log_message(f"[{data_type.upper()}] 기존 LabelEncoder를 사용하여 transform했습니다.")
                except ValueError as e:
                    log_message(f"치명적 오류: 검증 데이터에 훈련 데이터에 없는 클래스가 포함되어 있습니다. {repr(e)}")
                    return None, None, label_encoder
            else:
                 log_message("치명적 오류: 검증 데이터는 훈련 인코더가 필요합니다. 처리를 중단합니다.")
                 return None, None, label_encoder
        
        X_final = x_new_raw
        # Y_final은 위에서 이미 계산됨

    # 2-4. 저장 (NPY 파일 덮어쓰기)
    os.makedirs(x_npy_path.parent, exist_ok=True)
    
    np.save(x_npy_path, X_final)
    np.save(y_npy_path, Y_final) 
    
    # NPY 저장 후, 매니페스트 키 파일도 업데이트 (전체 키로 덮어쓰기)
    if is_incremental_mode:
        # 🌟 새로 처리된 키만 기존 파일에 추가 ('a' 모드)
        with open(processed_manifest_keys_path, 'a', encoding='utf-8') as f:
            for key in new_keys_to_process:
                f.write(key + '\n')
        log_message(f"✅ 처리 완료 키 파일 업데이트 완료: {processed_manifest_keys_path.name} ({len(new_keys_to_process)}개 추가)")
        
        # NPY 파일의 수정 시간을 매니페스트와 동기화
        os.utime(x_npy_path, (time.time(), os.path.getmtime(manifest_path)))
        os.utime(y_npy_path, (time.time(), os.path.getmtime(manifest_path)))

    elif force_reprocess and X_final.size > 0:
        # 전체 재처리 후에는 현재 매니페스트의 모든 키로 파일 재작성 ('w' 모드)
        all_keys_in_manifest = get_all_manifest_keys(manifest_path)
        with open(processed_manifest_keys_path, 'w', encoding='utf-8') as f:
            for key in all_keys_in_manifest:
                f.write(key + '\n')
        log_message(f"✅ 전체 재처리 후 처리 완료 키 파일 재작성 완료: {processed_manifest_keys_path.name}")
        # NPY 파일의 수정 시간을 매니페스트와 동기화
        os.utime(x_npy_path, (time.time(), os.path.getmtime(manifest_path)))
        os.utime(y_npy_path, (time.time(), os.path.getmtime(manifest_path)))


    log_message(f"[{data_type.upper()}] 데이터셋 저장 완료: {x_npy_path.name}, {y_npy_path.name}")
    log_message(f"[{data_type.upper()}] 최종 생성된 데이터 shape: X={X_final.shape}, Y={Y_final.shape}")
    
    return X_final, Y_final, label_encoder


def prepare_and_load_datasets(
    force_reprocess: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
    """
    전체 데이터 전처리 파이프라인을 실행하여 훈련 및 검증 데이터셋을 준비하고 로드합니다.
    """
    log_message("\n" + "#" * 50)
    log_message("### 데이터 전처리 파이프라인 시작 ###")
    log_message(f"강제 재처리 모드 (FORCE_REPROCESS): {force_reprocess}")
    
    # 0. 폴더 생성 확인
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ENCODER_PATH.parent, exist_ok=True)
    
    # 1. 초기화 (force_reprocess 인수를 사용)
    if force_reprocess:
        log_message("강제 재처리 모드 활성화: 기존 NPY 및 인코더 파일 초기화.")
        TRAIN_X_NPY_PATH.unlink(missing_ok=True)
        TRAIN_Y_NPY_PATH.unlink(missing_ok=True)
        VAL_X_NPY_PATH.unlink(missing_ok=True)
        VAL_Y_NPY_PATH.unlink(missing_ok=True)
        ENCODER_PATH.unlink(missing_ok=True)
        # 매니페스트 처리 기록 파일도 초기화
        PROCESSED_MANIFEST_TRAIN.unlink(missing_ok=True)
        PROCESSED_MANIFEST_VAL.unlink(missing_ok=True)

    
    # 2. 훈련 데이터 처리 
    X_train, Y_train, encoder = _load_and_preprocess_single_dataset(
        data_type='train',
        manifest_path=TRAIN_MANIFEST_PATH,
        x_npy_path=TRAIN_X_NPY_PATH,
        y_npy_path=TRAIN_Y_NPY_PATH,
        processed_manifest_keys_path=PROCESSED_MANIFEST_TRAIN, # ✅ 처리 키 경로 전달
        label_encoder=None,
        force_reprocess=force_reprocess
    )
    
    # 3. 인코더 저장 및 로드
    if encoder is not None:
        # NPY 파일이 새로 생성된 경우에만 인코더를 저장
        if not ENCODER_PATH.exists() or force_reprocess: 
            with open(ENCODER_PATH, 'w') as f:
                json.dump({'classes': list(encoder.classes_)}, f)
            log_message(f"최종 LabelEncoder 저장 완료: {ENCODER_PATH.name}, 클래스 수={len(encoder.classes_)}")
    elif ENCODER_PATH.exists():
        # NPY 파일을 재처리하지 않고 기존 파일을 로드한 경우, 인코더를 로드해야 함
        try:
            with open(ENCODER_PATH, 'r') as f:
                classes = json.load(f)['classes']
            encoder = LabelEncoder()
            encoder.classes_ = np.array(classes)
            log_message(f"기존 LabelEncoder 로드 완료. 클래스 수={len(encoder.classes_)}")
        except Exception as e:
            log_message(f"경고: 기존 인코더 로드 실패 ({repr(e)}). 검증 데이터 처리를 건너뜁니다.")
            encoder = None


    # 4. 검증 데이터 처리 
    X_val, Y_val = None, None
    if encoder is not None:
        X_val, Y_val, _ = _load_and_preprocess_single_dataset(
            data_type='val',
            manifest_path=VAL_MANIFEST_PATH,
            x_npy_path=VAL_X_NPY_PATH,
            y_npy_path=VAL_Y_NPY_PATH,
            processed_manifest_keys_path=PROCESSED_MANIFEST_VAL, # ✅ 처리 키 경로 전달
            label_encoder=encoder, # 훈련 데이터 인코더 전달
            force_reprocess=force_reprocess
        )
    else:
        log_message("LabelEncoder가 없으므로 검증 데이터 처리를 건너뜁니다.")

    
    # 5. 최종 요약
    log_message("\n" + "="*50)
    log_message("최종 데이터 로드 및 전처리 요약")
    if X_train is not None:
        log_message(f"훈련 데이터 (X_train): {X_train.shape}")
    if X_val is not None:
        log_message(f"검증 데이터 (X_val): {X_val.shape}")
    if encoder is not None:
        log_message(f"총 클래스 수: {len(encoder.classes_)}")
    log_message("### 데이터 전처리 파이프라인 완료 ###")
    log_message("#" * 50)
    
    return X_train, Y_train, X_val, Y_val, encoder
