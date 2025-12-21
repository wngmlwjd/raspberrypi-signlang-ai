import os
import glob
import json
from typing import List

def find_jsons_with_words(
        target_dirs: List[str], 
        target_words: List[str]
    ) -> List[str]:

        target_words = set(target_words)  # 빠른 검색을 위해 set 변환
        matched_jsons = []

        for root_dir in target_dirs:
            json_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)

            for json_path in json_files:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except:
                    continue

                segments = data.get("data", [])
                if not segments:
                    continue

                # JSON 내부 segment 검사
                for seg in segments:
                    attrs = seg.get("attributes")
                    if not attrs:
                        continue

                    word = attrs[0].get("name")
                    if not word:
                        continue

                    # 여러 단어 중 하나라도 매칭되면 통과
                    if word in target_words:
                        matched_jsons.append(json_path)
                        break  # 이 JSON에서 찾았으면 다음 JSON으로 이동

        return matched_jsons

if __name__ == "__main__":
    TARGET_DIRS = [
        "/Users/wngmlwjd/workspace/github/raspberrypi-signlang-ai/dataset/수어 영상/1.Training/morpheme/01",
    ]

    target_words = ["봐주다"]
    # target_words = ["꺼지다", "함구", "결점", "유구무언", "막히다", "얕보다", "근근이", "뻔뻔", "가다", "결심", "지도", "무자비", "놀랍다", "판박이", "맥없다", "욕하다", "대출", "아부", "봐주다", "빌리다", "주다", "아하", "내뱉다", "지불하다", "순식간", "격노", "포기", "맞다", "의문", "무례"]

    result = find_jsons_with_words(TARGET_DIRS, target_words)

    print("=== Found JSON files ===")
    for path in result:
        print(path)
