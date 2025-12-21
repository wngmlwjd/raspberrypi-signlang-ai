import os

from landmark_extractor.config import PREPROCESSED_DIR
from landmark_extractor.single_extract import extract_landmarks_from_single_video
from preprocess.single_preprocessing import preprocess_single_video
from test.single_test import run_single_test

TEST_VIDEO_PATH = "/Users/wngmlwjd/workspace/github/raspberrypi-signlang-ai/dataset/processed/single/raw/NIA_SL_WORD1501_REAL01_D.mp4"
DATE = "20251208_1"

extract_landmarks_from_single_video(TEST_VIDEO_PATH)

features = preprocess_single_video()

run_single_test(DATE)