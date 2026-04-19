import cv2
import numpy as np
import os
from src.detection.hand_detector import HandDetector
from src.features.feature_extractor import FeatureExtractor

class DataCollector:
    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
    
    def collect(self, word, person_id, num_samples=10, save_dir='data/raw/collected'):
        word_dir = os.path.join(save_dir, f'word_{word}')
        os.makedirs(word_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        print(f"开始采集 '{word}' 的数据，按 's' 保存，按 'q' 退出")
        
        sample_count = 0
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detector.detect(frame)
            frame = self.detector.draw_landmarks(frame, results)
            
            cv2.putText(frame, f'Word: {word}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Samples: {sample_count}/{num_samples}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Press s to save, q to quit', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                landmarks = self.detector.get_landmarks(results, frame.shape)
                if len(landmarks) > 0:
                    features = self.extractor.extract_features(landmarks)
                    file_name = f'{person_id}_{sample_count + 1:02d}.npy'
                    save_path = os.path.join(word_dir, file_name)
                    np.save(save_path, features)
                    print(f'Saved: {save_path}')
                    sample_count += 1
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    collector = DataCollector()
    word = input('请输入要采集的词语: ')
    person_id = input('请输入采集者ID: ')
    num_samples = int(input('请输入采集样本数: '))
    collector.collect(word, person_id, num_samples)