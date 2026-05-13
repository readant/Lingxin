"""
Preprocessor - 数据预处理工具

本文件实现了手语数据的预处理功能，将原始采集的数据转换为训练所需格式。

预处理步骤：
1. 从原始数据目录加载关键点序列
2. 对特征进行标准化处理
3. 保存处理后的特征矩阵和标签

输出文件：
- X.npy / X_sequence.npy: 特征矩阵
- y.npy / y_sequence.npy: 标签向量
- class_labels.npy: 类别标签映射字典

使用方法：
>>> python tools/preprocess.py
"""

import os
import numpy as np
from src.utils.data_loader import DataLoader
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    数据预处理器

    负责将原始采集的手语数据转换为训练所需的格式。
    提供两种预处理模式：
    - 普通特征：用于 SVM、RF、MLP 等传统机器学习模型
    - 序列特征：用于 LSTM、Transformer 等深度学习模型
    """

    def __init__(self):
        """初始化预处理器"""
        self.loader = DataLoader()
        self.scaler = StandardScaler()

    def preprocess(self, input_dir, output_dir):
        """
        预处理普通特征数据

        将关键点序列聚合为单一样本，进行标准化后保存。
        适用于传统机器学习模型。

        Args:
            input_dir (str): 原始数据目录，包含按词汇分类的.npy文件
            output_dir (str): 输出目录，保存预处理后的数据
        """
        os.makedirs(output_dir, exist_ok=True)

        # 加载原始数据
        X, y, class_labels = self.loader.load_data(input_dir)

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 保存预处理后的数据
        np.save(os.path.join(output_dir, 'X.npy'), X_scaled)
        np.save(os.path.join(output_dir, 'y.npy'), y)
        np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)

        print(f'预处理完成！数据已保存到 {output_dir}')
        print(f'样本数量: {len(X)}')
        print(f'类别数量: {len(class_labels)}')

    def preprocess_sequence(self, input_dir, output_dir, max_length=30):
        """
        预处理序列特征数据

        保持关键点序列的时序结构，进行标准化后保存。
        适用于深度学习模型。

        Args:
            input_dir (str): 原始数据目录
            output_dir (str): 输出目录
            max_length (int, optional): 最大序列长度. Defaults to 30.
                - 超过此长度的序列会被截断
                - 不足此长度的序列会进行padding
        """
        os.makedirs(output_dir, exist_ok=True)

        # 加载序列数据（自动进行padding/truncation）
        X, y, class_labels = self.loader.load_sequence_data(input_dir, max_length)

        # 对每个特征维度分别进行标准化
        # 注意：不能直接对整个矩阵进行标准化，因为序列长度不同
        for i in range(X.shape[2]):
            X[:, :, i] = self.scaler.fit_transform(X[:, :, i])

        # 保存预处理后的数据
        np.save(os.path.join(output_dir, 'X_sequence.npy'), X)
        np.save(os.path.join(output_dir, 'y_sequence.npy'), y)
        np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)

        print(f'序列数据预处理完成！数据已保存到 {output_dir}')
        print(f'样本数量: {len(X)}')
        print(f'序列长度: {max_length}')
        print(f'类别数量: {len(class_labels)}')


if __name__ == '__main__':
    preprocessor = Preprocessor()

    # 设置输入输出目录
    input_dir = 'data/raw/collected'
    output_dir = 'data/processed/csl_isolated'

    # 预处理普通特征数据
    print('=' * 50)
    print('预处理普通特征数据')
    print('=' * 50)
    preprocessor.preprocess(input_dir, output_dir)

    # 预处理序列特征数据
    print()
    print('=' * 50)
    print('预处理序列特征数据')
    print('=' * 50)
    preprocessor.preprocess_sequence(input_dir, output_dir, max_length=30)