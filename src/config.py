"""
统一配置管理

本文件集中管理项目中的所有路径、超参数和配置项，
避免在各模块中硬编码，提供单一配置来源。

使用示例：
>>> from src.config import config
>>> data = np.load(config.processed_data_dir / 'X.npy')
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class ProjectConfig:
    """
    项目全局配置

    所有路径均相对于项目根目录。
    可通过环境变量或直接赋值覆盖默认值。
    """

    # =========================================================================
    # 项目根目录
    # =========================================================================
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.resolve())

    # =========================================================================
    # 数据路径
    # =========================================================================
    data_dir: Path = field(default=None)           # 初始化后自动设置
    raw_data_dir: Path = field(default=None)
    processed_data_dir: Path = field(default=None)
    vocab_path: Path = field(default=None)

    # =========================================================================
    # 模型路径
    # =========================================================================
    models_dir: Path = field(default=None)
    mediapipe_models_dir: Path = field(default=None)

    # =========================================================================
    # 特征维度
    # =========================================================================
    raw_hand_landmarks: int = 21
    raw_hand_dims: int = 3
    raw_pose_landmarks: int = 15
    raw_pose_dims: int = 3
    extracted_feature_dims: int = 71

    # =========================================================================
    # 序列参数
    # =========================================================================
    max_sequence_length: int = 30
    min_sequence_length: int = 15
    max_raw_sequence_length: int = 150

    # =========================================================================
    # 训练参数
    # =========================================================================
    default_epochs: int = 50
    default_learning_rate: float = 0.001
    default_batch_size: int = 32
    test_split_ratio: float = 0.2
    early_stopping_patience: int = 10
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5

    # =========================================================================
    # 采集参数
    # =========================================================================
    default_target_samples: int = 30
    countdown_seconds: int = 3
    camera_index: int = 0

    # =========================================================================
    # 推理参数
    # =========================================================================
    inference_fps: int = 30

    # =========================================================================
    # 支持的模型类型
    # =========================================================================
    classifier_models: Tuple[str, ...] = ('svm', 'rf', 'mlp')
    deep_learning_models: Tuple[str, ...] = ('lstm', 'transformer')
    all_models: Tuple[str, ...] = ('svm', 'rf', 'mlp', 'lstm', 'transformer')

    # =========================================================================
    # 设备
    # =========================================================================
    device: str = 'cpu'  # 'cpu', 'cuda', 或 'auto'

    def __post_init__(self):
        """初始化派生路径"""
        # 数据路径
        if self.data_dir is None:
            self.data_dir = self.project_root / 'data'
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / 'raw' / 'collected'
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / 'processed' / 'csl_isolated'
        if self.vocab_path is None:
            self.vocab_path = self.data_dir / 'vocab.csv'

        # 模型路径
        if self.models_dir is None:
            self.models_dir = self.project_root / 'models'
        if self.mediapipe_models_dir is None:
            self.mediapipe_models_dir = self.models_dir

        # 自动检测设备
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def ensure_dirs(self):
        """确保所有必要的目录存在"""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_type: str) -> Path:
        """
        根据模型类型获取模型保存路径

        Args:
            model_type: 模型类型 ('svm', 'rf', 'mlp', 'lstm', 'transformer')

        Returns:
            Path: 模型文件路径
        """
        if model_type in self.classifier_models:
            return self.models_dir / f'{model_type}_model.pkl'
        elif model_type in self.deep_learning_models:
            return self.models_dir / f'{model_type}_model.pth'
        else:
            raise ValueError(f'未知模型类型: {model_type}')

    def get_class_labels_path(self) -> Path:
        """获取类别标签文件路径"""
        return self.processed_data_dir / 'class_labels.npy'

    def get_data_files(self, model_category: str) -> Dict[str, str]:
        """
        根据模型类别获取数据文件名

        Args:
            model_category: 'classifier' 或 'deep_learning'

        Returns:
            dict: {'X': ..., 'y': ...}
        """
        if model_category == 'classifier':
            return {'X': 'X.npy', 'y': 'y.npy'}
        else:
            return {'X': 'X_sequence.npy', 'y': 'y_sequence.npy'}


# 全局单例
config = ProjectConfig()
