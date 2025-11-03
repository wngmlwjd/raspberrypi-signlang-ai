"""
손 추적 및 랜드마크 추출 코드
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from src.mediapipe.config import MEDIAPIPE_HANDS_CONFIG

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, config=MEDIAPIPE_HANDS_CONFIG):
        self.hands = mp_hands.Hands(
            static_image_mode=config["static_image_mode"],
            max_num_hands=config["max_num_hands"],
            min_detection_confidence=config["min_detection_confidence"],
            min_tracking_confidence=config["min_tracking_confidence"]
        )
        
    def process_image(self, image_path):
        """
        이미지 경로를 받아 손 관절 랜드마크를 반환
        """
        
        image = cv2.imread(image_path)

        if image is None:
            raise IOError(f"Cannot read image {image_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        # 랜드마크 추출 로직은 그대로 유지
        return results.multi_hand_landmarks
    
    def save_landmarks(self, landmarks, save_path):
        """
        landmarks: list of 손 랜드마크 (x, y, z) 좌표
        save_path: 저장할 파일 경로 (ex: .npy)
        """
        
        if landmarks is None:
            # 손 인식 실패 시 빈 배열 저장 가능
            np.save(save_path, np.array([]))
            return
        
        all_hands_coords = []
        for hand_landmarks in landmarks:
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            all_hands_coords.append(coords)

        np.save(save_path, np.array(all_hands_coords))
    
    def draw_and_save_landmarks(self, image_path, save_path, landmarks):
        """
        이미지에 손 랜드마크를 그려 반환
        """
        
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Cannot load image from {image_path}")
        
        if landmarks:
            for hand_landmarks in landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imwrite(save_path, image)
        
    def close(self):
        """
        MediaPipe Hands 리소스를 해제합니다.
        """
        self.hands.close()
    
def process_frames(frame_dir):
    """
    프레임 이미지 폴더를 순회하며 손 관절 정보 추출
    실행 예시 함수
    """
    
    tracker = HandTracker()
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        landmarks = tracker.process_image(frame_path)
        
        print(f"{frame_file} landmarks count: {len(landmarks)}")
