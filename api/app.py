"""
Flask API 服务 — 手语识别 REST API (带 WebSocket 实时推理)

端点：
    GET  /               — 首页
    GET  /demo           — 演示页
    POST /api/predict    — 单帧预测
    POST /api/load_model — 加载模型
    GET  /api/health     — 健康检查
    WS   /ws/detect      — WebSocket 实时检测+预测
"""

import sys
import os
import json
import base64
import numpy as np
import cv2
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

from src.detection.hand_detector import HolisticDetector
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)

# =============================================================================
# 全局状态
# =============================================================================
detector = None
model = None
scaler = None
class_labels = None
class_names = None
model_type = None
_model_loaded = False
sequence_buffer = []
MAX_SEQ_LENGTH = 30


def get_detector():
    global detector
    if detector is None:
        detector = HolisticDetector(min_detection_confidence=0.3)
        logger.info("HolisticDetector 初始化完成")
    return detector


def load_model(m_type='lstm', m_path=None):
    global model, scaler, class_labels, class_names, model_type, _model_loaded
    model_type = m_type

    labels_path = str(config.processed_data_dir / 'class_labels.npy')
    if os.path.exists(labels_path):
        class_labels = np.load(labels_path, allow_pickle=True).item()
        class_names = list(class_labels.keys())
    else:
        return False

    num_classes = len(class_names)

    if m_path is None:
        m_path = str(config.get_model_path(m_type))

    if not os.path.exists(m_path):
        return False

    if m_type in ('svm', 'rf', 'mlp'):
        model_data = joblib.load(m_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
    elif m_type in ('lstm', 'transformer'):
        import torch
        if m_type == 'lstm':
            from src.models.lstm_model import LSTMModel
            model = LSTMModel((30, 171), num_classes)
        else:
            from src.models.transformer_model import TransformerModel
            model = TransformerModel((30, 171), num_classes)
        model.load(m_path)
        model.eval()

        scaler_path = str(config.processed_data_dir / 'scaler_sequence.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None

    _model_loaded = True
    logger.info(f"模型加载成功: {m_type} ({num_classes} 类)")
    return True


def detect_and_predict(landmarks_171):
    """对 171 维特征进行预测"""
    global sequence_buffer

    if not _model_loaded or model is None:
        return None, 0.0

    if model_type in ('svm', 'rf', 'mlp'):
        features = np.array(landmarks_171, dtype=np.float32).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)
        pred = model.predict(features)[0]
        pred_idx = int(pred)
        if hasattr(model, 'predict_proba'):
            confidence = float(model.predict_proba(features)[0][pred_idx])
        else:
            confidence = 1.0
        return class_names[pred_idx], confidence

    else:
        sequence_buffer.append(landmarks_171)
        if len(sequence_buffer) > MAX_SEQ_LENGTH:
            sequence_buffer = sequence_buffer[-MAX_SEQ_LENGTH:]

        if len(sequence_buffer) < MAX_SEQ_LENGTH:
            return None, 0.0

        import torch
        seq = np.array([sequence_buffer], dtype=np.float32)
        if scaler is not None:
            n_batch, seq_len, n_feat = seq.shape
            seq_2d = seq.reshape(-1, n_feat)
            seq_scaled = scaler.transform(seq_2d)
            seq = seq_scaled.reshape(n_batch, seq_len, n_feat)

        input_tensor = torch.tensor(seq, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        return class_names[pred_idx], confidence


def extract_landmarks_from_frame(frame):
    """从帧中提取 171 维关键点"""
    # 镜像帧（与训练时采集数据一致）
    frame = cv2.flip(frame, 1)

    det = get_detector()
    results = det.detect(frame)
    landmarks = det.get_landmarks(results, frame.shape)

    # 归一化 x,y 到 [0,1]
    h, w = frame.shape[:2]
    landmarks_norm = landmarks.copy()
    landmarks_norm[0::3] /= w
    landmarks_norm[1::3] /= h

    has_hand = bool(np.any(np.abs(landmarks[:126]) > 1e-4))
    return landmarks_norm.tolist(), has_hand


# =============================================================================
# REST API
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """接收 base64 图片或 171 维特征向量进行预测"""
    try:
        data = request.json
        if data is None:
            return jsonify({'error': '空请求体'}), 400

        # 接收 base64 图片
        if 'image' in data:
            img_bytes = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'error': '图片解码失败'}), 400
            landmarks, has_hand = extract_landmarks_from_frame(frame)
        # 接收 171 维特征
        elif 'landmarks' in data:
            landmarks = data['landmarks']
            has_hand = bool(np.any(np.abs(np.array(landmarks)[:126]) > 1e-4))
        else:
            return jsonify({'error': '请提供 image 或 landmarks'}), 400

        if not has_hand:
            return jsonify({
                'prediction': None,
                'confidence': 0.0,
                'has_hand': False,
                'buffer_size': len(sequence_buffer),
                'buffer_target': MAX_SEQ_LENGTH
            })

        prediction, confidence = detect_and_predict(landmarks)

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'has_hand': True,
            'buffer_size': len(sequence_buffer),
            'buffer_target': MAX_SEQ_LENGTH
        })

    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """仅检测手部关键点（不预测），返回归一化的 171 维特征"""
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': '需要 image 字段'}), 400

        img_bytes = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': '图片解码失败'}), 400

        landmarks, has_hand = extract_landmarks_from_frame(frame)

        return jsonify({
            'landmarks': landmarks,
            'has_hand': has_hand
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    data = request.json or {}
    m_type = data.get('model_type', 'lstm')
    m_path = data.get('model_path')
    success = load_model(m_type, m_path)
    if success:
        global sequence_buffer
        sequence_buffer = []
        return jsonify({'status': 'ok', 'num_classes': len(class_names), 'model_type': m_type})
    return jsonify({'status': 'error'})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': _model_loaded,
        'model_type': model_type,
        'num_classes': len(class_names) if class_names else 0,
        'has_scaler': scaler is not None,
        'has_detector': detector is not None
    })


@app.route('/api/collect', methods=['POST'])
def api_collect():
    """启动数据采集器（后台进程）"""
    import subprocess
    import threading

    data = request.json or {}
    person_id = data.get('person_id', '')
    target_samples = data.get('target_samples', 30)

    if not person_id:
        return jsonify({'error': '请提供录制人ID'}), 400

    def run_collector():
        try:
            cmd = [
                sys.executable, 'tools/collect_data.py',
                '--vocab', 'data/vocab.csv',
                '--output', 'data/raw/collected'
            ]
            env = os.environ.copy()
            env['LINGXIN_PERSON_ID'] = person_id
            env['LINGXIN_TARGET_SAMPLES'] = str(target_samples)

            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            logger.info(f"采集器已启动: PID={process.pid}, 人员={person_id}")
        except Exception as e:
            logger.error(f"启动采集器失败: {e}")

    thread = threading.Thread(target=run_collector, daemon=True)
    thread.start()

    return jsonify({
        'status': 'ok',
        'message': f'采集器已启动，录制人: {person_id}'
    })


@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    """启动数据预处理"""
    import subprocess
    import threading

    def run_preprocess():
        try:
            cmd = [
                sys.executable, 'tools/preprocess.py',
                '--input', 'data/raw/collected',
                '--output', 'data/processed'
            ]
            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            logger.info(f"预处理已启动: PID={process.pid}")
        except Exception as e:
            logger.error(f"启动预处理失败: {e}")

    thread = threading.Thread(target=run_preprocess, daemon=True)
    thread.start()

    return jsonify({'status': 'ok', 'message': '预处理已启动'})


@app.route('/api/train', methods=['POST'])
def api_train():
    """启动模型训练"""
    import subprocess
    import threading

    data = request.json or {}
    model_type = data.get('model_type', 'lstm')
    epochs = data.get('epochs', 50)

    def run_train():
        try:
            cmd = [
                sys.executable, 'tools/train.py',
                '--model', model_type,
                '--epochs', str(epochs)
            ]
            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            logger.info(f"训练已启动: PID={process.pid}, 模型={model_type}")
        except Exception as e:
            logger.error(f"启动训练失败: {e}")

    thread = threading.Thread(target=run_train, daemon=True)
    thread.start()

    return jsonify({'status': 'ok', 'message': f'训练已启动: {model_type}'})


# =============================================================================
# 静态页面
# =============================================================================

@app.route('/')
def index():
    return send_from_directory('../web', 'index_new.html')

@app.route('/old')
def index_old():
    return send_from_directory('../web', 'index.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('../web', 'dashboard.html')

@app.route('/demo.html')
def demo_html():
    return send_from_directory('../web', 'demo.html')

@app.route('/resources')
def resources():
    return send_from_directory('../web', 'resources.html')

@app.route('/docs-viewer.html')
def docs_viewer():
    return send_from_directory('../web', 'docs-viewer.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('../assets', 'logo.png')


# =============================================================================
# 静态资源（替代 static_folder）
# =============================================================================

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """静态资源（图片、JS 等）"""
    return send_from_directory('../assets', filename)

@app.route('/static/css/<path:filename>')
def serve_css(filename):
    """CSS 样式文件"""
    return send_from_directory('../web/static/css', filename)

@app.route('/static/js/<path:filename>')
def serve_js(filename):
    """JavaScript 文件"""
    return send_from_directory('../web/static/js', filename)

@app.route('/docs/<path:filename>')
def serve_docs(filename):
    """Markdown 文档"""
    return send_from_directory('../docs', filename)

@app.route('/README.md')
def serve_readme():
    return send_from_directory('..', 'README.md')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--model', default='lstm', choices=['svm', 'rf', 'mlp', 'lstm', 'transformer'])
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    load_model(args.model, args.model_path)
    get_detector()

    logger.info(f"启动 http://{args.host}:{args.port}")
    app.run(debug=args.debug, host=args.host, port=args.port)
