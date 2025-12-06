import os
import json
from collections import defaultdict

from preprocess.config import (
    MORPHEMES_DIR,          # 예: [".../morpheme/train", ".../morpheme/valid"]
    TRAINING_VIDEOS_LIST,   # 예: [1, 2, 3, 4]
    VALIDATION_VIDEOS_LIST, # 예: [17, 18]
    LABEL_LIST_PATH,        # 결과 저장 파일
)


def extract_labels_from_folder(folder_path):
    """
    해당 폴더 안의 모든 *_morpheme.json 파일을 읽고
    label(name)을 set 형태로 반환.
    """
    labels = set()

    if not os.path.isdir(folder_path):
        return labels

    for file in os.listdir(folder_path):
        if not file.endswith("_morpheme.json"):
            continue

        json_path = os.path.join(folder_path, file)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        if "data" not in data:
            continue

        for seg in data["data"]:
            if "attributes" in seg:
                for attr in seg["attributes"]:
                    name = attr.get("name")
                    if name:
                        labels.add(name)

    return labels



def collect_speaker_labels(base_dir, speaker_ids):
    """
    - speaker_ids 에 해당하는 폴더만 사용
    - 화자별 label set 생성
    - 화자별 label 등장 영상 수 카운트
    - 화자별 json 파일 수 반환 (총 영상 수 계산용)
    """
    speaker_label_sets = {}
    speaker_label_counts = {}
    speaker_video_counts = {}

    for spk in speaker_ids:
        folder_name = f"{spk:02d}"
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        labels = extract_labels_from_folder(folder_path)
        speaker_label_sets[spk] = labels

        spk_counts = defaultdict(int)
        video_count = 0

        for file in os.listdir(folder_path):
            if not file.endswith("_morpheme.json"):
                continue

            video_count += 1
            json_path = os.path.join(folder_path, file)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue

            json_labels = set()
            if "data" in data:
                for seg in data["data"]:
                    if "attributes" in seg:
                        for attr in seg["attributes"]:
                            name = attr.get("name")
                            if name:
                                json_labels.add(name)

            # 영상 하나에 등장한 class count 증가
            for lbl in json_labels:
                spk_counts[lbl] += 1

        speaker_label_counts[spk] = dict(spk_counts)
        speaker_video_counts[spk] = video_count

    return speaker_label_sets, speaker_label_counts, speaker_video_counts



def generate_labels_list():
    """
    공통 label 처리, label별 영상수 계산, labels_list.txt 저장
    + Training / Validation / 전체 영상 개수 출력
    """

    # -------- Training --------
    train_label_sets, train_label_counts, train_video_counts = collect_speaker_labels(
        MORPHEMES_DIR[0], TRAINING_VIDEOS_LIST
    )

    # -------- Validation --------
    val_label_sets, val_label_counts, val_video_counts = collect_speaker_labels(
        MORPHEMES_DIR[1], VALIDATION_VIDEOS_LIST
    )

    # -------- 통합 --------
    all_speakers = list(train_label_sets.keys()) + list(val_label_sets.keys())
    all_label_sets = {**train_label_sets, **val_label_sets}
    all_label_counts = {**train_label_counts, **val_label_counts}

    # 영상 개수 통합
    total_train_videos = sum(train_video_counts.values())
    total_valid_videos = sum(val_video_counts.values())
    total_processed_videos = total_train_videos + total_valid_videos

    # -------- 공통 label 추출 --------
    if not all_label_sets:
        print("No valid speakers found.")
        return

    common_labels = None
    for spk in all_label_sets:
        if common_labels is None:
            common_labels = set(all_label_sets[spk])
        else:
            common_labels &= all_label_sets[spk]

    if common_labels is None:
        common_labels = set()

    # -------- label별 전체 영상 count --------
    label_total_count = defaultdict(int)
    for lbl in common_labels:
        for spk in all_label_counts:
            label_total_count[lbl] += all_label_counts[spk].get(lbl, 0)

    # 정렬
    sorted_labels = sorted(label_total_count.items(), key=lambda x: x[1], reverse=True)

    # -------- 저장 --------
    os.makedirs(os.path.dirname(LABEL_LIST_PATH), exist_ok=True)

    with open(LABEL_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("# label speaker_counts total_count\n")

        for lbl, total in sorted_labels:
            per_speaker_counts = [all_label_counts[spk].get(lbl, 0) for spk in all_speakers]
            per_speaker_str = " ".join(str(x) for x in per_speaker_counts)
            f.write(f"{lbl} {per_speaker_str} {total}\n")

    # -------- 결과 출력 --------
    print(f"[완료] labels_list.txt 생성 (총 {len(sorted_labels)}개 클래스)")
    print("---------------")
    print(f"[Training 영상 수]     {total_train_videos} 개")
    print(f"[Validation 영상 수]  {total_valid_videos} 개")
    print(f"[전체 처리한 영상 수] {total_processed_videos} 개")
    print("---------------")


if __name__ == "__main__":
    generate_labels_list()
