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
from flask_sock import Sock

from src.detection.hand_detector import HolisticDetector
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)
sock = Sock(app)

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
                'landmarks': landmarks,
                'buffer_size': len(sequence_buffer),
                'buffer_target': MAX_SEQ_LENGTH
            })

        prediction, confidence = detect_and_predict(landmarks)

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'has_hand': True,
            'landmarks': landmarks,
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


@app.route('/api/data/stats', methods=['GET'])
def api_data_stats():
    """获取录制数据统计信息"""
    import csv

    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'collected')
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')

    # 读取词汇表
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'vocab.csv')
    vocab = []
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            vocab = list(reader)

    # 统计原始数据
    raw_stats = []
    total_raw = 0
    if os.path.exists(raw_dir):
        for word_dir in sorted(os.listdir(raw_dir)):
            word_path = os.path.join(raw_dir, word_dir)
            if os.path.isdir(word_path):
                npy_files = [f for f in os.listdir(word_path) if f.endswith('.npy')]
                count = len(npy_files)
                total_raw += count
                raw_stats.append({
                    'word': word_dir,
                    'count': count,
                    'category': next((v['category'] for v in vocab if v['word'] == word_dir), '')
                })

    # 检查处理后数据
    processed_exists = os.path.exists(os.path.join(processed_dir, 'X.npy'))
    processed_seq_exists = os.path.exists(os.path.join(processed_dir, 'X_sequence.npy'))

    # 检查模型
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    trained_models = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(('.pkl', '.pth')) and not f.startswith('hand_') and not f.startswith('pose_'):
                trained_models.append(f)

    return jsonify({
        'status': 'ok',
        'vocab_count': len(vocab),
        'total_raw_samples': total_raw,
        'word_count': len(raw_stats),
        'raw_stats': raw_stats,
        'processed_exists': processed_exists,
        'processed_seq_exists': processed_seq_exists,
        'trained_models': trained_models,
        'model_count': len(trained_models)
    })


@app.route('/api/data/delete', methods=['POST'])
def api_data_delete():
    """删除指定词汇的录制数据"""
    data = request.json or {}
    word = data.get('word', '')

    if not word:
        return jsonify({'error': '请指定要删除的词汇'}), 400

    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'collected', word)

    if not os.path.exists(raw_dir):
        return jsonify({'error': f'词汇 {word} 的数据目录不存在'}), 404

    import shutil
    try:
        file_count = len([f for f in os.listdir(raw_dir) if f.endswith('.npy')])
        shutil.rmtree(raw_dir)
        logger.info(f"已删除词汇数据: {word} ({file_count}个样本)")
        return jsonify({'status': 'ok', 'message': f'已删除 {word} 的 {file_count} 个样本'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/delete_all', methods=['POST'])
def api_data_delete_all():
    """删除所有录制数据（危险操作）"""
    data = request.json or {}
    confirm = data.get('confirm', False)

    if not confirm:
        return jsonify({'error': '请确认删除所有数据'}), 400

    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'collected')

    if not os.path.exists(raw_dir):
        return jsonify({'error': '数据目录不存在'}), 404

    import shutil
    try:
        total_files = 0
        for word_dir in os.listdir(raw_dir):
            word_path = os.path.join(raw_dir, word_dir)
            if os.path.isdir(word_path):
                total_files += len([f for f in os.listdir(word_path) if f.endswith('.npy')])
                shutil.rmtree(word_path)
        logger.info(f"已删除所有录制数据: {total_files}个样本")
        return jsonify({'status': 'ok', 'message': f'已删除所有数据 ({total_files}个样本)'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# WebSocket 实时检测+预测
# =============================================================================

@sock.route('/ws/detect')
def ws_detect(ws):
    """WebSocket 实时检测+预测

    接收: JSON {"image": "base64_string"}
    返回: JSON {"has_hand": bool, "predictions": [...], "buffer_size": int, "buffer_target": int}
    """
    global sequence_buffer

    logger.info("WebSocket 客户端已连接")

    while True:
        try:
            message = ws.receive()
            if message is None:
                break

            data = json.loads(message)

            if 'image' not in data:
                ws.send(json.dumps({'error': '需要 image 字段'}))
                continue

            # 解码图片
            img_bytes = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                ws.send(json.dumps({'error': '图片解码失败'}))
                continue

            # 提取关键点
            landmarks, has_hand = extract_landmarks_from_frame(frame)

            if not has_hand:
                ws.send(json.dumps({
                    'has_hand': False,
                    'predictions': [],
                    'buffer_size': len(sequence_buffer),
                    'buffer_target': MAX_SEQ_LENGTH
                }))
                continue

            # 预测
            if _model_loaded and model is not None:
                if model_type in ('svm', 'rf', 'mlp'):
                    features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                    if scaler is not None:
                        features = scaler.transform(features)
                    pred = model.predict(features)[0]
                    pred_idx = int(pred)
                    if hasattr(model, 'predict_proba'):
                        confidence = float(model.predict_proba(features)[0][pred_idx])
                    else:
                        confidence = 1.0
                    predictions = [{'word': class_names[pred_idx], 'confidence': confidence, 'category': ''}]
                else:
                    # 深度学习模型需要序列
                    sequence_buffer.append(landmarks)
                    if len(sequence_buffer) > MAX_SEQ_LENGTH:
                        sequence_buffer = sequence_buffer[-MAX_SEQ_LENGTH:]

                    if len(sequence_buffer) >= MAX_SEQ_LENGTH:
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

                        # 返回 top-5 预测
                        top_indices = np.argsort(probs)[::-1][:5]
                        predictions = [
                            {'word': class_names[i], 'confidence': float(probs[i]), 'category': ''}
                            for i in top_indices
                        ]
                    else:
                        predictions = []

                ws.send(json.dumps({
                    'has_hand': True,
                    'predictions': predictions,
                    'buffer_size': len(sequence_buffer),
                    'buffer_target': MAX_SEQ_LENGTH
                }))
            else:
                ws.send(json.dumps({
                    'has_hand': True,
                    'predictions': [],
                    'buffer_size': len(sequence_buffer),
                    'buffer_target': MAX_SEQ_LENGTH,
                    'warning': '模型未加载'
                }))

        except Exception as e:
            logger.error(f"WebSocket 处理错误: {e}", exc_info=True)
            try:
                ws.send(json.dumps({'error': str(e)}))
            except:
                break

    logger.info("WebSocket 客户端已断开")


# =============================================================================
# 模型管理 API
# =============================================================================

@app.route('/api/models', methods=['GET'])
def api_models_list():
    """列出所有已训练的模型"""
    models_dir = str(config.models_dir)
    models = []

    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(('.pkl', '.pth')) and not f.startswith('hand_') and not f.startswith('pose_'):
                file_path = os.path.join(models_dir, f)
                stat = os.stat(file_path)

                # 确定模型类型
                if f.endswith('.pkl'):
                    m_type = f.replace('_model.pkl', '').upper()
                else:
                    m_type = f.replace('_model.pth', '').upper()

                models.append({
                    'name': f,
                    'type': m_type,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'loaded': (model_type is not None and
                              f == config.get_model_path(model_type).name and
                              _model_loaded)
                })

    return jsonify({'status': 'ok', 'models': models})


@app.route('/api/models/load', methods=['POST'])
def api_models_load():
    """加载指定模型"""
    data = request.json or {}
    model_name = data.get('model_name', '')

    if not model_name:
        return jsonify({'error': '请指定模型名称'}), 400

    # 从文件名推断模型类型
    if model_name.endswith('_model.pkl'):
        m_type = model_name.replace('_model.pkl', '')
    elif model_name.endswith('_model.pth'):
        m_type = model_name.replace('_model.pth', '')
    else:
        return jsonify({'error': '无法识别模型类型'}), 400

    m_path = str(config.models_dir / model_name)
    success = load_model(m_type, m_path)

    if success:
        global sequence_buffer
        sequence_buffer = []
        return jsonify({'status': 'ok', 'model_type': m_type, 'num_classes': len(class_names)})
    return jsonify({'status': 'error', 'message': '模型加载失败'})


@app.route('/api/models/delete', methods=['POST'])
def api_models_delete():
    """删除指定模型"""
    data = request.json or {}
    model_name = data.get('model_name', '')

    if not model_name:
        return jsonify({'error': '请指定模型名称'}), 400

    model_path = str(config.models_dir / model_name)

    if not os.path.exists(model_path):
        return jsonify({'error': f'模型 {model_name} 不存在'}), 404

    try:
        os.remove(model_path)
        logger.info(f"已删除模型: {model_name}")
        return jsonify({'status': 'ok', 'message': f'已删除模型 {model_name}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 词汇表 API
# =============================================================================

@app.route('/api/vocab', methods=['GET'])
def api_vocab():
    """获取词汇表"""
    import csv

    vocab_path = str(config.vocab_path)
    vocab = []

    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            vocab = list(reader)

    return jsonify({'status': 'ok', 'vocab': vocab, 'count': len(vocab)})


# =============================================================================
# 模型评估 API
# =============================================================================

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """启动模型评估"""
    import subprocess
    import threading

    data = request.json or {}
    model_type = data.get('model_type', 'lstm')
    model_path = data.get('model_path')

    def run_evaluate():
        try:
            cmd = [
                sys.executable, 'tools/evaluate.py',
                '--model', model_type
            ]
            if model_path:
                cmd.extend(['--checkpoint', model_path])

            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            stdout, stderr = process.communicate()
            logger.info(f"评估完成: {model_type}, 返回码={process.returncode}")
            if stdout:
                logger.info(f"评估输出: {stdout.decode('utf-8', errors='replace')}")
            if stderr:
                logger.warning(f"评估错误: {stderr.decode('utf-8', errors='replace')}")
        except Exception as e:
            logger.error(f"启动评估失败: {e}")

    thread = threading.Thread(target=run_evaluate, daemon=True)
    thread.start()

    return jsonify({'status': 'ok', 'message': f'评估已启动: {model_type}'})


@app.route('/api/evaluate/result', methods=['GET'])
def api_evaluate_result():
    """获取最新的评估结果"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'data', 'processed')
    results_file = os.path.join(results_dir, 'evaluation_results.json')

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return jsonify({'status': 'ok', 'results': results})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    return jsonify({'status': 'ok', 'results': None, 'message': '暂无评估结果'})


# =============================================================================
# 静态页面
# =============================================================================

@app.route('/')
def index():
    return send_from_directory('../web', 'index.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('../web', 'dashboard.html')

@app.route('/demo')
@app.route('/demo.html')
def demo():
    return send_from_directory('../web', 'demo.html')

@app.route('/docs')
@app.route('/docs.html')
def docs():
    return send_from_directory('../web', 'docs.html')

@app.route('/resources')
def resources():
    return send_from_directory('../web', 'resources.html')

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

    print("\n" + "="*50)
    print("  聆心手语识别系统 - API服务已启动")
    print("="*50)
    print(f"\n  访问地址: http://localhost:{args.port}")
    print(f"\n  页面链接:")
    print(f"    http://localhost:{args.port}/              首页")
    print(f"    http://localhost:{args.port}/dashboard     控制台")
    print(f"    http://localhost:{args.port}/demo          实时演示")
    print(f"    http://localhost:{args.port}/docs          文档")
    print(f"    http://localhost:{args.port}/resources     资源")
    print(f"\n  API端点:")
    print(f"    http://localhost:{args.port}/api/health    健康检查")
    print(f"    http://localhost:{args.port}/api/predict   预测接口")
    print(f"    http://localhost:{args.port}/api/models    模型列表")
    print("="*50 + "\n")

    # 使用 werkzeug 内置服务器（支持 WebSocket）
    from werkzeug.serving import make_server
    server = make_server(args.host, args.port, app, threaded=True)
    server.serve_forever()
