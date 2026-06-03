"""
TrainRunner - 训练入口

本文件是整个项目的训练入口，提供统一的命令行接口用于训练不同类型的模型。

功能特性：
- 支持多种模型类型：SVM、RF、MLP、LSTM、Transformer
- 自动选择数据加载方式
- 自动保存训练好的模型
- 输出训练准确率

使用方法：
python tools/train.py
然后按照提示输入模型类型。

示例：
请选择模型类型 (svm/rf/mlp/lstm/transformer): lstm
"""

import numpy as np
import os
from src.training.trainer import Trainer
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.config import config
from src.utils.logger import get_logger


class TrainRunner:
    """
    训练运行器

    负责解析命令行输入，调用对应的训练方法，并保存模型。
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    # 模型配置映射
    MODEL_MAP = {
        'svm': {'type': 'classifier', 'model_class': None},
        'rf': {'type': 'classifier', 'model_class': None},
        'mlp': {'type': 'classifier', 'model_class': None},
        'lstm': {'type': 'deep_learning', 'model_class': LSTMModel},
        'transformer': {'type': 'deep_learning', 'model_class': TransformerModel}
    }

    def run(self, data_dir=None, model_type='svm', save_dir=None):
        """
        执行训练流程

        Args:
            data_dir (str, optional): 数据目录路径. 默认使用 config.processed_data_dir.
            model_type (str, optional): 模型类型. Defaults to 'svm'.
            save_dir (str, optional): 模型保存目录. 默认使用 config.models_dir.
        """
        # 使用配置中的默认路径
        if data_dir is None:
            data_dir = str(config.processed_data_dir)
        if save_dir is None:
            save_dir = str(config.models_dir)

        # 创建模型保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 检查模型类型是否有效
        if model_type not in self.MODEL_MAP:
            valid_types = ", ".join(self.MODEL_MAP.keys())
            self.logger.error(f'未知模型类型: {model_type}，可选类型: {valid_types}')
            return

        # 获取模型配置
        model_info = self.MODEL_MAP[model_type]
        model_type_category = model_info['type']
        data_files = config.get_data_files(model_type_category)

        # 加载数据
        self.logger.info(f'正在加载 {model_type_category} 数据...')
        X = np.load(os.path.join(data_dir, data_files['X']))
        y = np.load(os.path.join(data_dir, data_files['y']))

        self.logger.info(f'数据加载完成！样本数: {X.shape[0]}')
        self.logger.info(f'特征形状: {X.shape[1:]}')

        # 根据模型类别选择训练方法
        if model_type_category == 'classifier':
            self.logger.info(f'正在训练 {model_type} 分类器...')
            model_data, accuracy = Trainer.train_classifier(X, y, model_type)

            # 保存模型（使用joblib）
            model_path = config.get_model_path(model_type)
            Trainer.save_classifier(model_data, str(model_path))
            self.logger.info(f'分类器模型已保存到: {model_path}')
        else:
            model_class = model_info['model_class']
            self.logger.info(f'正在训练 {model_type} 模型...')
            model, accuracy = Trainer.train_deep_learning(X, y, model_class)

            # 保存模型（使用PyTorch）
            model_path = config.get_model_path(model_type)
            model.save(str(model_path))
            self.logger.info(f'深度学习模型已保存到: {model_path}')

        # 输出训练结果
        self.logger.info(f'{model_type} 模型训练完成！准确率: {accuracy:.4f}')


if __name__ == '__main__':
    runner = TrainRunner()

    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')

    runner.run(model_type=model_type)
