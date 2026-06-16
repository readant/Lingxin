"""
InferenceRunner - 实时推理运行器

支持的模型类型：
- 传统机器学习：SVM、随机森林、MLP
- 深度学习：LSTM、Transformer

使用方法：
python tools/inference.py --model lstm
"""

import os
import sys
import cv2
import numpy as np
import argparse
import joblib
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.hand_detector import HolisticDetector
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.config import config
from src.utils.logger import get_logger

logger = get_logger("InferenceRunner")


def init_font(size=24):
    """加载中文字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                continue
    return ImageFont.load_default()


def put_chinese_text(frame, text, position, font, color=(255, 255, 255)):
    """在帧上绘制中文文本"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=tuple(reversed(color)))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class InferenceRunner:

    MODEL_CONFIG = {
        'svm': {'category': 'classifier', 'uses_sequence': False},
        'rf': {'category': 'classifier', 'uses_sequence': False},
        'mlp': {'category': 'classifier', 'uses_sequence': False},
        'lstm': {'category': 'deep_learning', 'uses_sequence': True},
        'transformer': {'category': 'deep_learning', 'uses_sequence': True},
    }

    def __init__(self):
        self.detector = HolisticDetector(min_detection_confidence=0.3)
        self.sequence_buffer = []
        self.max_sequence_length = 30
        self.scaler = None
        self.font = init_font(26)
        self.font_small = init_font(18)

    def run(self, model_type, model_path=None, class_labels_path=None):
        if model_type not in self.MODEL_CONFIG:
            logger.error(f"未知模型类型: {model_type}")
            return

        if class_labels_path is None:
            class_labels_path = str(config.processed_data_dir / 'class_labels.npy')

        config_info = self.MODEL_CONFIG[model_type]
        category = config_info['category']
        uses_sequence = config_info['uses_sequence']

        # 加载类别标签
        class_labels = np.load(class_labels_path, allow_pickle=True).item()
        class_names = list(class_labels.keys())
        num_classes = len(class_labels)
        logger.info(f"加载 {num_classes} 个类别")

        # 加载模型
        if category == 'classifier':
            model = self._load_classifier(model_path, model_type)
        else:
            model = self._load_deep_learning(model_type, model_path, num_classes)

        # 打开摄像头
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error("错误：无法打开摄像头")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 预热
        for _ in range(15):
            cap.grab()

        logger.info(f"开始 {model_type} 实时推理，按 Q/ESC 退出")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # 检测关键点
                results = self.detector.detect(frame)
                landmarks = self.detector.get_landmarks(results, frame.shape)
                frame = self.detector.draw_landmarks(frame, results)

                # 归一化 x,y 到 [0,1]（与训练时一致）
                h, w = frame.shape[:2]
                landmarks_norm = landmarks.copy()
                landmarks_norm[0::3] /= w
                landmarks_norm[1::3] /= h

                has_hand = bool(np.any(np.abs(landmarks[:126]) > 1e-4))

                prediction = None
                if has_hand:
                    if uses_sequence:
                        prediction = self._predict_sequence(model, landmarks_norm, class_names)
                    else:
                        prediction = self._predict_single(model, landmarks_norm, class_names)

                # 绘制预测结果（用 PIL 渲染中文）
                if prediction:
                    frame = put_chinese_text(
                        frame, f"识别: {prediction}",
                        (10, 10), self.font, (0, 255, 0))

                # 状态
                if has_hand:
                    frame = put_chinese_text(
                        frame, "已检测到手部",
                        (10, 50), self.font_small, (0, 255, 0))
                else:
                    frame = put_chinese_text(
                        frame, "请将手伸到摄像头前",
                        (10, 50), self.font_small, (0, 0, 255))

                # 缓冲区信息
                if uses_sequence:
                    buf_text = f"缓冲: {len(self.sequence_buffer)}/{self.max_sequence_length}"
                    frame = put_chinese_text(
                        frame, buf_text,
                        (10, 80), self.font_small, (255, 255, 255))

                # 模型信息
                frame = put_chinese_text(
                    frame, f"模型: {model_type.upper()} | 按Q退出",
                    (10, h - 30), self.font_small, (200, 200, 200))

                cv2.imshow('Sign Language Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _load_classifier(self, model_path, model_type):
        if model_path is None:
            model_path = str(config.get_model_path(model_type))
        model_data = joblib.load(model_path)
        self.scaler = model_data.get('scaler')
        logger.info(f"加载分类器: {model_path}")
        return model_data

    def _load_deep_learning(self, model_type, model_path, num_classes):
        import torch
        model_classes = {'lstm': LSTMModel, 'transformer': TransformerModel}
        model_class = model_classes.get(model_type)
        if model_class is None:
            raise ValueError(f"未知模型类型: {model_type}")

        input_shape = (self.max_sequence_length, 171)
        model = model_class(input_shape, num_classes)

        if model_path is None:
            model_path = str(config.get_model_path(model_type))

        model.load(model_path)

        scaler_path = str(config.processed_data_dir / 'scaler_sequence.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"加载 scaler: {scaler_path}")

        logger.info(f"加载模型: {model_path}")
        return model

    def _predict_single(self, model, landmarks, class_names):
        features = landmarks.reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        pred = model['model'].predict(features)[0]
        return class_names[int(pred)]

    def _predict_sequence(self, model, landmarks, class_names):
        self.sequence_buffer.append(landmarks)
        if len(self.sequence_buffer) > self.max_sequence_length:
            self.sequence_buffer = self.sequence_buffer[-self.max_sequence_length:]

        if len(self.sequence_buffer) < self.max_sequence_length:
            return None

        import torch
        input_data = np.array(self.sequence_buffer, dtype=np.float32)

        if self.scaler is not None:
            seq_len, n_features = input_data.shape
            input_2d = input_data.reshape(-1, n_features)
            input_scaled = self.scaler.transform(input_2d)
            input_data = input_scaled.reshape(seq_len, n_features)

        input_tensor = torch.tensor(input_data).unsqueeze(0).to(model.device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_idx = predicted.item()

        return class_names[pred_idx]


def main():
    parser = argparse.ArgumentParser(description='聆心手语识别 — 实时推理')
    parser.add_argument('--model', type=str, required=True,
                       choices=['svm', 'rf', 'mlp', 'lstm', 'transformer'],
                       help='模型类型')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)
    args = parser.parse_args()

    runner = InferenceRunner()
    runner.run(args.model, args.model_path, args.labels)


if __name__ == '__main__':
    main()
