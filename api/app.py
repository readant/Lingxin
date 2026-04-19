from flask import Flask, request, jsonify
import numpy as np
from src.detection.hand_detector import HandDetector
from src.features.feature_extractor import FeatureExtractor
from src.models.classifiers import Classifiers
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel

app = Flask(__name__)

detector = HandDetector()
extractor = FeatureExtractor()

class_labels = np.load('data/processed/csl_isolated/class_labels.npy', allow_pickle=True).item()
class_names = list(class_labels.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        landmarks = np.array(data['landmarks'])
        
        features = extractor.extract_features(landmarks)
        
        # 这里需要加载训练好的模型
        # 暂时返回模拟结果
        prediction = np.random.randint(0, len(class_names))
        predicted_class = class_names[prediction]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': 0.95
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)