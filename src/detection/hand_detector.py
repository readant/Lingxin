import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results
    
    def get_landmarks(self, results, image_shape):
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmark = []
                for lm in hand_landmarks.landmark:
                    x = lm.x * image_shape[1]
                    y = lm.y * image_shape[0]
                    z = lm.z
                    hand_landmark.append([x, y, z])
                landmarks.append(hand_landmark)
        return np.array(landmarks)
    
    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return image
    
    def close(self):
        self.hands.close()