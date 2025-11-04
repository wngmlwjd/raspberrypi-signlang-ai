from pathlib import Path

from src.mediapipe.preprocess_for_train import process_videos
from dataset.create_manifest import run_manifest_generation
from sign_language_recognition.train.preprocess import prepare_and_load_datasets
from sign_language_recognition.train.train import train_sign_language_model
from sign_language_recognition.test.evaluate import evaluate_model_for_paper

from sign_language_recognition.train.utils import log_message

def mediapipe_video_processing(N=None):
    # --- 실행 환경 설정 ---
    ROOT_VIDEO_DIR = "dataset/수어 영상/1.Training" # 원본 비디오 파일 경로
    LANDMARKS_BASE_DIR = "dataset/processed/landmarks"          # 랜드마크 저장 경로
    DRAW_IMG_BASE_DIR = "dataset/processed/drawings"           # (선택 사항) 결과 이미지 저장 경로

    # Reprocess Mode: PROCESSED 목록에 있지만 NPY 파일이 없는 영상을 찾아 재처리합니다.
    # 일반적인 신규 처리는 reprocess_mode=False로 실행하거나, 파일을 찾아서 제거한 후 실행합니다.
    REPROCESS_MISSING_MODE = True 

    process_videos(
        root_video_dir=ROOT_VIDEO_DIR,
        draw_img_base_dir=DRAW_IMG_BASE_DIR,
        landmarks_base_dir=LANDMARKS_BASE_DIR,
        max_count=N,
        reprocess_mode=REPROCESS_MISSING_MODE
    )

    
if __name__ == "__main__":
    # MediaPipe를 이용한 영상 처리 test
    # mediapipe_video_processing()
    # mediapipe_video_processing(500)
    '''
    10개 = 약 1분
    100개 = 약 9분
    200개 = 약 19분
    220개 = 약 16분
    300개 = 약 29분
    390개 = 약 37분
    500개 = 약 45분
    2000개 = 약 3시간 50분
    2560개 = 약 5시간 50분
    5200개 = 약 10시간
    10000개 = 약 16시간
    '''
    
    # 매니페스트에 새 데이터 추가 test
    # new_entries_train, new_entries_val = run_manifest_generation(reset_files=False)
    # print(f"매니페스트 생성 완료. 훈련 항목: {new_entries_train}, 검증 항목: {new_entries_val}")
    
    # # 이 파일 단독 실행 시 데이터 생성만 테스트
    # # force_reprocess = True면 기존 데이터 무시하고 새로 생성
    # prepare_and_load_datasets(force_reprocess=False)
    
    # 모델 학습 test
    # train 데이터 -> train, val 데이터 -> val, test 분리(validation_split)
    # retrain=True면 새 모델 생성 후 학습, False면 기존 모델 불러와 추가 학습
    history, X_test_final, Y_test_final, model_save_path = train_sign_language_model(epochs=10, batch_size=256, validation_split=0.5, retrain=False)
    # epochs=1 = 약 5분
    
    # 성능 평가 및 그래프 추출
    if X_test_final.shape[0] > 0:
        evaluate_model_for_paper(
            model_path=str(model_save_path), # 훈련에서 저장된 모델 경로
            X_test=X_test_final,              # 훈련에서 분리된 최종 테스트 X 데이터
            y_test=Y_test_final               # 훈련에서 분리된 최종 테스트 Y 데이터
        )
    else:
        log_message("🚨 최종 테스트 데이터가 없어 성능 평가를 건너뜁니다.")