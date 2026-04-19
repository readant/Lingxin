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

class PoseDetector:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results
    
    def get_landmarks(self, results, image_shape):
        landmarks = []
        if results.pose_landmarks:
            for i in range(15):  # 只提取上半身15个关键点（索引0~14）
                lm = results.pose_landmarks.landmark[i]
                x = lm.x * image_shape[1]
                y = lm.y * image_shape[0]
                z = lm.z
                landmarks.append([x, y, z])
        return np.array(landmarks)
    
    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return image
    
    def close(self):
        self.pose.close()

class HolisticDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hand_detector = HandDetector(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.pose_detector = PoseDetector(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect(self, image):
        hand_results = self.hand_detector.detect(image)
        pose_results = self.pose_detector.detect(image)
        return hand_results, pose_results
    
    def get_landmarks(self, results, image_shape):
        hand_results, pose_results = results
        
        # 处理手部关键点
        hand_landmarks = self.hand_detector.get_landmarks(hand_results, image_shape)
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))
        
        if len(hand_landmarks) > 0:
            # 假设第一个是左手，第二个是右手
            if len(hand_landmarks) >= 1:
                left_hand = hand_landmarks[0]
            if len(hand_landmarks) >= 2:
                right_hand = hand_landmarks[1]
        
        # 处理姿态关键点
        pose_landmarks = self.pose_detector.get_landmarks(pose_results, image_shape)
        if len(pose_landmarks) == 0:
            pose_landmarks = np.zeros((15, 3))
        
        # 组合为171维向量
        combined = np.concatenate([
            left_hand.flatten(),
            right_hand.flatten(),
            pose_landmarks.flatten()
        ])
        return combined
    
    def draw_landmarks(self, image, results):
        hand_results, pose_results = results
        image = self.hand_detector.draw_landmarks(image, hand_results)
        image = self.pose_detector.draw_landmarks(image, pose_results)
        return image
    
    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()