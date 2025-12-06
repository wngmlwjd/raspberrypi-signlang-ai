import os
from typing import Set

def load_words_from_file(file_path: str) -> Set[str]:
    """
    텍스트 파일에서 단어를 읽어와 집합(Set) 형태로 반환합니다.
    (공백을 제거하고, 빈 줄은 무시하며, 대소문자는 구분하여 처리)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"오류: 파일을 찾을 수 없습니다. 경로: '{file_path}'")
    
    words = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 양 끝 공백 (줄 바꿈 문자 포함) 제거
            word = line.strip()
            # 단어가 비어있지 않다면 집합에 추가
            if word:
                words.add(word)
    return words

def check_inclusion(file_a_path: str, file_b_path: str):
    """
    File A의 모든 단어가 File B에 포함되는지 확인합니다.
    
    Args:
        file_a_path: 부분 집합이 될 것으로 예상되는 파일 (체크할 단어 목록)
        file_b_path: 전체 집합이 될 것으로 예상되는 파일 (포함 여부를 확인할 단어 목록)
    """
    print(f"--- 파일 포함 여부 비교 시작 ---")
    print(f"체크 대상 (File A): {file_a_path}")
    print(f"기준 파일 (File B): {file_b_path}\n")

    try:
        # 1. 파일에서 단어 목록 로드 (Set으로 로드하여 빠른 비교 준비)
        words_a = load_words_from_file(file_a_path)
        words_b = load_words_from_file(file_b_path)
        
        if not words_a:
            print(f"✅ 결과: {file_a_path}이 비어있으므로, 논리적으로 포함 관계로 간주됩니다.")
            return

        # 2. 포함 관계 확인 (A가 B의 부분 집합인지 확인)
        # issubset() 메서드는 Set A의 모든 요소가 Set B에 포함되어 있는지 확인합니다.
        is_subset = words_a.issubset(words_b)

        if is_subset:
            print(f"🎉 포함됨: '{file_a_path}'의 **모든** 단어가 '{file_b_path}'에 포함됩니다.")
        else:
            print(f"❌ 불포함: '{file_a_path}'의 일부 단어가 '{file_b_path}'에 누락되었습니다.")
            
            # 3. 누락된 단어 확인 및 출력
            # A - B 연산은 A에는 있지만 B에는 없는 단어를 찾습니다.
            missing_words = words_a - words_b
            if missing_words:
                print(f"\n--- 누락된 단어 ({len(missing_words)}개) ---")
                print(", ".join(sorted(missing_words)))
                print("---------------------------------------")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"처리 중 예상치 못한 오류 발생: {e}")

# --- 실행 부분 ---
# 비교할 두 파일의 경로를 설정합니다.
FILE_A = './dataset/labels/labels_30.txt' # 부분 집합이 될 단어 파일
FILE_B = './dataset/labels/label_list.txt' # 전체 목록 단어 파일

# 예시 파일 생성 (실제 파일 경로로 대체하여 사용하세요)
# create_example_files(FILE_A, FILE_B) # 예시 파일을 사용하려면 주석 해제

check_inclusion(FILE_A, FILE_B)