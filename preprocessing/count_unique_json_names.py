import os
import json

from preprocessing.config import (
    MORPHEMES_DIR,
    TRAINING_VIDEOS_LIST,
    VALIDATION_VIDEOS_LIST,
    LABEL_LIST_PATH,
)


def extract_labels_from_folder(folder_path):
    """
    주어진 morpheme 폴더(예: .../morpheme/01)에 포함된
    모든 JSON 파일에서 label(name)들을 추출하여 set으로 반환
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


def get_common_labels(base_dir, video_num_list):
    """
    morphemes_dir(예: /1.Training/morpheme) 아래에 있는
    video_num_list(예: [1,2,3]) 폴더들에서
    공통으로 등장하는 label만 반환
    """
    common = None  # 처음에는 None → 첫 폴더의 label set으로 설정

    for num in video_num_list:
        folder = f"{num:02d}"
        folder_path = os.path.join(base_dir, folder)

        labels = extract_labels_from_folder(folder_path)

        if common is None:
            common = labels
        else:
            common = common & labels  # 교집합

    return common if common is not None else set()


def generate_labels_list():
    """
    Training·Validation 각각에서 videos_list 폴더들에 공통으로 있는 label만 추출하고,
    최종적으로 두 집합의 교집합만 labels_list.txt로 저장
    """
    # Training
    training_common = get_common_labels(MORPHEMES_DIR[0], TRAINING_VIDEOS_LIST)

    # Validation
    validation_common = get_common_labels(MORPHEMES_DIR[1], VALIDATION_VIDEOS_LIST)

    # 최종 교집합
    final_labels = sorted(training_common & validation_common)

    # 저장
    os.makedirs(os.path.dirname(LABEL_LIST_PATH), exist_ok=True)
    with open(LABEL_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(final_labels))

    print(f"labels_list.txt 생성 완료 ({len(final_labels)} labels)")


if __name__ == "__main__":
    generate_labels_list()
