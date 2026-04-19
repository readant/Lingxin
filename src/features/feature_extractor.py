import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, landmarks):
        features = []
        for hand_landmarks in landmarks:
            if len(hand_landmarks) == 0:
                continue
            
            wrist = hand_landmarks[0]
            relative_landmarks = hand_landmarks - wrist
            
            palm_center = np.mean(hand_landmarks[0:5], axis=0)
            fingers = [
                hand_landmarks[5:9],
                hand_landmarks[9:13],
                hand_landmarks[13:17],
                hand_landmarks[17:21]
            ]
            
            finger_lengths = []
            for finger in fingers:
                length = np.linalg.norm(finger[-1] - finger[0])
                finger_lengths.append(length)
            
            angles = self.calculate_angles(hand_landmarks)
            
            hand_features = np.concatenate([
                relative_landmarks.flatten(),
                finger_lengths,
                angles
            ])
            features.append(hand_features)
        
        if not features:
            return np.zeros(63 + 4 + 4)
        
        return np.mean(features, axis=0)
    
    def calculate_angles(self, landmarks):
        angles = []
        fingers = [
            [0, 1, 2],
            [0, 5, 6],
            [0, 9, 10],
            [0, 13, 14]
        ]
        
        for finger in fingers:
            a = landmarks[finger[0]]
            b = landmarks[finger[1]]
            c = landmarks[finger[2]]
            
            v1 = a - b
            v2 = c - b
            
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(cosine_angle)
            angles.append(angle)
        
        return np.array(angles)