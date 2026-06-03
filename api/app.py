"""
Flask API 服务 — 手语识别 REST API

提供实时手语识别接口和静态页面服务。
启动方式：
    python api/app.py
    或
    cd api && python app.py

端点：
    POST /api/predict   — 手语识别预测
    GET  /api/health    — 健康检查
    GET  /              — 项目首页
    GET  /resources     — 资源导航页
"""

import sys
import os
import numpy as np

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.detection.hand_detector import HandDetector
from src.features.feature_extractor import FeatureExtractor
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Flask 应用初始化
# =============================================================================
app = Flask(
    __name__,
    static_folder='../',           # 项目根目录作为静态文件目录
    static_url_path=''
)
CORS(app)  # 允许跨域请求

# =============================================================================
# 模型和检测器初始化（懒加载）
# =============================================================================
detector = None
extractor = None
model = None
class_labels = None
class_names = None

_MODEL_LOADED = False


def _init_components():
    """延迟初始化检测器和特征提取器（首次请求时加载）"""
    global detector, extractor
    if detector is None:
        detector = HandDetector()
        logger.info("HandDetector 初始化完成")
    if extractor is None:
        extractor = FeatureExtractor()
        logger.info("FeatureExtractor 初始化完成")


def load_model(model_type: str = 'lstm', model_path: str = None):
    """
    加载训练好的模型和类别标签

    Args:
        model_type: 模型类型 ('lstm' 或 'transformer')
        model_path: 模型文件路径，None 则使用默认路径
    """
    global model, class_labels, class_names, _MODEL_LOADED

    _init_components()

    # 加载模型
    if model_path is None:
        model_path = str(config.get_model_path(model_type))

    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}，使用随机预测模式")
        _MODEL_LOADED = False
        return False

    # 加载类别标签
    labels_path = str(config.get_class_labels_path())
    if os.path.exists(labels_path):
        class_labels = np.load(labels_path, allow_pickle=True).item()
        class_names = list(class_labels.keys())
    else:
        logger.warning(f"类别标签文件不存在: {labels_path}")
        class_names = []
        _MODEL_LOADED = False
        return False

    # 创建并加载模型
    if model_type == 'lstm':
        from src.models.lstm_model import LSTMModel
        model = LSTMModel(input_size=config.extracted_feature_dims,
                          hidden_size=128, num_layers=2,
                          num_classes=len(class_names))
    elif model_type == 'transformer':
        from src.models.transformer_model import TransformerModel
        model = TransformerModel(input_size=config.extracted_feature_dims,
                                 num_classes=len(class_names))
    else:
        logger.error(f"不支持的模型类型: {model_type}")
        return False

    model.load(model_path)
    model.eval()
    _MODEL_LOADED = True
    logger.info(f"模型加载成功: {model_path} ({len(class_names)} 个类别)")

    return True


# =============================================================================
# API 端点
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    手语识别预测

    请求体 (JSON):
        {
            "landmarks": [[x1,y1,z1], [x2,y2,z2], ...],  # 手部关键点 (21, 3)
            "sequence": [[...], [...], ...]                # 或序列数据 (30, 71)
        }

    响应 (JSON):
        {
            "prediction": "你好",
            "confidence": 0.95,
            "model_loaded": true
        }
    """
    try:
        _init_components()
        data = request.json

        if data is None:
            return jsonify({'error': '请求体不能为空'}), 400

        # 支持两种输入：单帧关键点 or 序列特征
        if 'sequence' in data:
            # 序列输入（深度学习模型）
            sequence = np.array(data['sequence'], dtype=np.float32)
            if sequence.ndim == 2:
                sequence = np.expand_dims(sequence, axis=0)  # (1, seq_len, features)
        elif 'landmarks' in data:
            # 关键点输入
            landmarks = np.array(data['landmarks'])
            features = extractor.extract_features([landmarks])
            sequence = np.expand_dims(features, axis=(0, 1))  # (1, 1, features)
        else:
            return jsonify({'error': '请提供 landmarks 或 sequence 字段'}), 400

        # 推理
        if _MODEL_LOADED and model is not None:
            import torch
            with torch.no_grad():
                input_tensor = torch.tensor(sequence, dtype=torch.float32)
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                prediction = class_names[pred_idx]
        else:
            # 模型未加载时的降级处理
            if class_names:
                prediction = class_names[0]
                confidence = 0.0
            else:
                return jsonify({
                    'error': '模型未加载，请先调用 /api/load_model',
                    'prediction': None,
                    'confidence': 0.0,
                    'model_loaded': False
                }), 503

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'model_loaded': _MODEL_LOADED
        })

    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """
    加载/切换模型

    请求体 (JSON):
        {
            "model_type": "lstm",          # 可选，默认 'lstm'
            "model_path": "/path/to/model"  # 可选，默认使用配置路径
        }
    """
    try:
        data = request.json or {}
        model_type = data.get('model_type', 'lstm')
        model_path = data.get('model_path')

        success = load_model(model_type, model_path)

        if success:
            return jsonify({
                'status': 'ok',
                'message': f'{model_type} 模型加载成功',
                'num_classes': len(class_names)
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': f'模型文件不存在: {model_path or config.get_model_path(model_type)}'
            })

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': _MODEL_LOADED,
        'num_classes': len(class_names) if class_names else 0
    })


# =============================================================================
# 静态页面路由
# =============================================================================

@app.route('/')
def index():
    """项目首页"""
    return send_from_directory('..', 'index.html')


@app.route('/resources')
def resources():
    """资源导航页"""
    return send_from_directory('..', 'resources.html')


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='聆心手语识别 API 服务')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='监听端口 (默认: 5000)')
    parser.add_argument('--model', default='lstm', choices=['lstm', 'transformer'],
                        help='预加载模型类型 (默认: lstm)')
    parser.add_argument('--model-path', default=None, help='模型文件路径 (可选)')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')

    args = parser.parse_args()

    # 尝试预加载模型
    if args.model_path or args.model:
        load_model(args.model, args.model_path)

    logger.info(f"启动 API 服务 http://{args.host}:{args.port}")
    app.run(debug=args.debug, host=args.host, port=args.port)
