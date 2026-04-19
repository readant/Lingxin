import cv2
import numpy as np
from src.detection.hand_detector import HandDetector
from src.features.feature_extractor import FeatureExtractor
from src.models.classifiers import Classifiers
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel

class InferenceRunner:
    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
    
    def run(self, model_type='svm', model_path=None, class_labels_path='data/processed/csl_isolated/class_labels.npy'):
        class_labels = np.load(class_labels_path, allow_pickle=True).item()
        class_names = list(class_labels.keys())
        
        if model_type in ['svm', 'rf', 'mlp']:
            clf = Classifiers()
            # 这里需要加载训练好的模型
            # model = clf.get_model(model_type)
        elif model_type == 'lstm':
            model = LSTMModel((30, 71), len(class_labels))
            if model_path:
                model.load(model_path)
        elif model_type == 'transformer':
            model = TransformerModel((30, 71), len(class_labels))
            if model_path:
                model.load(model_path)
        else:
            print(f'Unknown model type: {model_type}')
            return
        
        cap = cv2.VideoCapture(0)
        print('开始实时推理，按 q 退出')
        
        sequence = []
        max_sequence_length = 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detector.detect(frame)
            frame = self.detector.draw_landmarks(frame, results)
            
            landmarks = self.detector.get_landmarks(results, frame.shape)
            if len(landmarks) > 0:
                features = self.extractor.extract_features(landmarks)
                sequence.append(features)
                
                if len(sequence) > max_sequence_length:
                    sequence = sequence[-max_sequence_length:]
                
                if len(sequence) == max_sequence_length:
                    input_data = np.array(sequence).reshape(1, max_sequence_length, -1)
                    if model_type in ['svm', 'rf', 'mlp']:
                        # y_pred = model.predict(input_data.reshape(1, -1))
                        pass
                    else:
                        y_pred = model.predict(input_data)
                        y_pred = np.argmax(y_pred, axis=1)[0]
                        predicted_class = class_names[y_pred]
                        
                        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Inference', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    runner = InferenceRunner()
    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')
    model_path = input('请输入模型路径 (可选): ')
    runner.run(model_type, model_path)