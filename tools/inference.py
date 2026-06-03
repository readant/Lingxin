"""
InferenceRunner - 实时推理运行器

本文件实现了手语识别的实时推理功能，通过摄像头捕获视频流并进行预测。

工作流程：
1. 从摄像头读取视频帧
2. 使用 MediaPipe 检测手部关键点
3. 提取特征向量
4. 输入模型进行预测
5. 显示预测结果

支持的模型类型：
- 传统机器学习：SVM、随机森林、MLP（使用单帧特征）
- 深度学习：LSTM、Transformer（使用序列特征）

使用方法：
python tools/inference.py
然后按照提示输入模型类型和模型路径。
"""

import cv2
import numpy as np
from src.detection.hand_detector import HandDetector
from src.features.feature_extractor import FeatureExtractor
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.utils.logger import get_logger


class InferenceRunner:
    """
    实时推理运行器

    负责初始化检测器、特征提取器和模型，并处理实时推理流程。
    使用字典映射消除 if-elif 分支，提高可维护性。
    """

    # 模型配置映射
    # 键: 模型类型名称
    # 值: 配置字典
    MODEL_CONFIG = {
        'svm': {
            'model_category': 'classifier',  # 模型类别
            'input_shape': 71,                # 单帧特征维度
            'uses_sequence': False,            # 是否使用序列
        },
        'rf': {
            'model_category': 'classifier',
            'input_shape': 71,
            'uses_sequence': False,
        },
        'mlp': {
            'model_category': 'classifier',
            'input_shape': 71,
            'uses_sequence': False,
        },
        'lstm': {
            'model_category': 'deep_learning',
            'input_shape': (30, 71),          # (序列长度, 特征维度)
            'uses_sequence': True,
        },
        'transformer': {
            'model_category': 'deep_learning',
            'input_shape': (30, 71),
            'uses_sequence': True,
        }
    }

    def __init__(self):
        """初始化推理运行器"""
        self.logger = get_logger(self.__class__.__name__)
        # 初始化检测器和特征提取器
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()

        # 推理序列缓冲区（用于LSTM/Transformer）
        self.sequence_buffer = []
        self.max_sequence_length = 30

    def run(self, model_type='svm', model_path=None,
            class_labels_path='data/processed/csl_isolated/class_labels.npy'):
        """
        执行实时推理

        Args:
            model_type (str, optional): 模型类型. Defaults to 'svm'.
            model_path (str, optional): 模型文件路径. Defaults to None.
            class_labels_path (str, optional): 类别标签文件路径. Defaults to '...'.
        """
        # 验证模型类型
        if model_type not in self.MODEL_CONFIG:
            self.logger.error(f'未知模型类型: {model_type}，可选类型: {", ".join(self.MODEL_CONFIG.keys())}')
            return

        config = self.MODEL_CONFIG[model_type]

        # 加载类别标签
        class_labels = np.load(class_labels_path, allow_pickle=True).item()
        class_names = list(class_labels.keys())
        num_classes = len(class_labels)

        # 创建模型
        model = self._create_model(model_type, config, num_classes)

        # 如果提供了模型路径，加载预训练权重
        if model_path:
            model.load(model_path)

        # 打开摄像头开始推理
        self._run_inference_loop(model, model_type, config, class_names)

    def _create_model(self, model_type, config, num_classes):
        """
        创建模型实例

        Args:
            model_type (str): 模型类型
            config (dict): 模型配置
            num_classes (int): 类别数量

        Returns:
            模型实例
        """
        model_category = config['model_category']

        if model_category == 'classifier':
            # 传统机器学习分类器（预留，当前版本未实现加载）
            raise NotImplementedError(
                f'{model_type} 分类器的推理功能尚未实现。'
                f'请使用深度学习模型（lstm/transformer）。'
            )
        else:
            # 深度学习模型
            model_class = self._get_model_class(model_type)
            if model_class is None:
                raise ValueError(f'未知模型类型: {model_type}')

            input_shape = config['input_shape']
            return model_class(input_shape, num_classes)

    def _get_model_class(self, model_type):
        """
        获取模型类

        Args:
            model_type (str): 模型类型

        Returns:
            class: 模型类
        """
        model_classes = {
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        return model_classes.get(model_type)

    def _run_inference_loop(self, model, model_type, config, class_names):
        """
        运行推理主循环

        Args:
            model: 模型实例
            model_type (str): 模型类型
            config (dict): 模型配置
            class_names (list): 类别名称列表
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error('错误：无法打开摄像头')
            return

        self.logger.info(f'开始 {model_type} 实时推理，按 q 退出')

        uses_sequence = config['uses_sequence']

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测手部关键点
            results = self.detector.detect(frame)
            frame = self.detector.draw_landmarks(frame, results)

            # 提取关键点
            landmarks = self.detector.get_landmarks(results, frame.shape)

            if len(landmarks) > 0:
                # 提取特征
                features = self.extractor.extract_features(landmarks)

                if uses_sequence:
                    # 序列模型：维护特征序列
                    self._handle_sequence_inference(
                        model, model_type, frame, features, class_names
                    )
                else:
                    # 分类器模型：单帧预测（预留）
                    self._handle_classifier_inference(frame)

            cv2.imshow('Inference', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _handle_sequence_inference(self, model, model_type, frame, features, class_names):
        """
        处理序列模型推理

        Args:
            model: 模型实例
            model_type (str): 模型类型
            frame: 当前帧
            features: 特征向量
            class_names (list): 类别名称列表
        """
        # 添加到序列缓冲区
        self.sequence_buffer.append(features)

        # 保持序列长度不超过最大值
        if len(self.sequence_buffer) > self.max_sequence_length:
            self.sequence_buffer = self.sequence_buffer[-self.max_sequence_length:]

        # 当序列达到最大长度时进行预测
        if len(self.sequence_buffer) == self.max_sequence_length:
            input_data = np.array(self.sequence_buffer).reshape(
                1, self.max_sequence_length, -1
            )

            # 模型预测
            y_pred = model.predict(input_data)
            predicted_class = class_names[int(y_pred[0])]

            # 在帧上显示预测结果
            cv2.putText(
                frame,
                f'Prediction: {predicted_class}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # 显示当前序列状态
            cv2.putText(
                frame,
                f'Sequence: {len(self.sequence_buffer)}/{self.max_sequence_length}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )

    def _handle_classifier_inference(self, frame):
        """
        处理分类器模型推理（预留功能）

        Args:
            frame: 当前帧
        """
        # 分类器推理功能预留
        # 当前版本尚未实现加载和推理分类器模型
        pass


if __name__ == '__main__':
    runner = InferenceRunner()

    # 获取用户输入
    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')
    model_path = input('请输入模型路径 (可选): ').strip() or None

    # 运行推理
    runner.run(model_type, model_path)
