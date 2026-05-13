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


class TrainRunner:
    """
    训练运行器
    
    负责解析命令行输入，调用对应的训练方法，并保存模型。
    """
    
    # 模型配置映射
    # 键: 模型类型名称
    # 值: {'type': 模型类别(classifier/deep_learning), 'model_class': 模型类}
    MODEL_MAP = {
        'svm': {'type': 'classifier', 'model_class': None},
        'rf': {'type': 'classifier', 'model_class': None},
        'mlp': {'type': 'classifier', 'model_class': None},
        'lstm': {'type': 'deep_learning', 'model_class': LSTMModel},
        'transformer': {'type': 'deep_learning', 'model_class': TransformerModel}
    }

    # 数据文件配置
    # 根据模型类别选择不同的数据文件
    DATA_FILES = {
        'classifier': {'X': 'X.npy', 'y': 'y.npy'},
        'deep_learning': {'X': 'X_sequence.npy', 'y': 'y_sequence.npy'}
    }

    def run(self, data_dir, model_type='svm', save_dir='models'):
        """
        执行训练流程
        
        Args:
            data_dir (str): 数据目录路径
            model_type (str, optional): 模型类型. Defaults to 'svm'.
            save_dir (str, optional): 模型保存目录. Defaults to 'models'.
        """
        # 创建模型保存目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 检查模型类型是否有效
        if model_type not in self.MODEL_MAP:
            print(f'未知模型类型: {model_type}，可选类型: {", ".join(self.MODEL_MAP.keys())}')
            return

        # 获取模型配置
        model_info = self.MODEL_MAP[model_type]
        model_type_category = model_info['type']
        data_files = self.DATA_FILES[model_type_category]

        # 加载数据
        print(f'正在加载 {model_type_category} 数据...')
        X = np.load(os.path.join(data_dir, data_files['X']))
        y = np.load(os.path.join(data_dir, data_files['y']))
        
        print(f'数据加载完成！样本数: {X.shape[0]}')
        print(f'特征形状: {X.shape[1:]}')

        # 根据模型类别选择训练方法
        if model_type_category == 'classifier':
            # 训练传统机器学习分类器
            print(f'正在训练 {model_type} 分类器...')
            model_data, accuracy = Trainer.train_classifier(X, y, model_type)
            
            # 保存模型（使用joblib）
            model_path = os.path.join(save_dir, f'{model_type}_model.pkl')
            Trainer.save_classifier(model_data, model_path)
            print(f'分类器模型已保存到: {model_path}')
        else:
            # 训练深度学习模型
            model_class = model_info['model_class']
            print(f'正在训练 {model_type} 模型...')
            model, accuracy = Trainer.train_deep_learning(X, y, model_class)
            
            # 保存模型（使用PyTorch）
            model_path = os.path.join(save_dir, f'{model_type}_model.h5')
            model.save(model_path)
            print(f'深度学习模型已保存到: {model_path}')

        # 输出训练结果
        print(f'{model_type} 模型训练完成！准确率: {accuracy:.4f}')


if __name__ == '__main__':
    # 创建训练运行器实例
    runner = TrainRunner()
    
    # 设置数据目录（假设数据已经预处理好）
    data_dir = 'data/processed/csl_isolated'
    
    # 从命令行获取模型类型
    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')
    
    # 执行训练
    runner.run(data_dir, model_type)