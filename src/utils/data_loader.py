"""
DataLoader - 数据加载器

本文件实现了手语数据的加载功能，支持两种数据格式：
1. 特征数据：用于传统机器学习模型（SVM、RF、MLP）
2. 序列数据：用于深度学习模型（LSTM、Transformer）

同时还提供标准 PyTorch Dataset 类，可直接用于 torch.utils.data.DataLoader。

数据目录结构要求：
data/
├── vocab.csv              # 词汇表
└── processed/
    └── csl_isolated/
        ├── 你/
        │   ├── sample1.npy
        │   ├── sample2.npy
        │   └── ...
        ├── 我/
        │   ├── sample1.npy
        │   └── ...
        └── ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class DataLoader:
    """
    手语数据加载器

    负责从磁盘加载预处理后的手语数据，支持特征数据和序列数据两种格式。
    """

    def __init__(self):
        """初始化数据加载器"""
        pass

    def load_data(self, data_dir):
        """
        加载特征数据（用于传统机器学习模型）

        Args:
            data_dir (str): 数据目录路径，包含各个词汇的子目录

        Returns:
            tuple: (X, y, class_labels)
                   X: 特征矩阵，形状为(n_samples, n_features)
                   y: 标签向量，形状为(n_samples,)
                   class_labels: 类别标签字典，{类别名: 类别索引}
        """
        X = []
        y = []
        class_labels = {}
        label_idx = 0

        for word_dir in sorted(os.listdir(data_dir)):
            word_path = os.path.join(data_dir, word_dir)

            if not os.path.isdir(word_path):
                continue

            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1

            for file_name in os.listdir(word_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(word_path, file_name)
                    data = np.load(file_path)
                    X.append(data)
                    y.append(class_labels[word_dir])

        X = np.array(X)
        y = np.array(y)

        return X, y, class_labels

    def load_sequence_data(self, data_dir, max_length=30):
        """
        加载序列数据（用于深度学习模型）

        Args:
            data_dir (str): 数据目录路径
            max_length (int, optional): 最大序列长度. Defaults to 30.

        Returns:
            tuple: (X, y, class_labels)
        """
        X = []
        y = []
        class_labels = {}
        label_idx = 0

        for word_dir in sorted(os.listdir(data_dir)):
            word_path = os.path.join(data_dir, word_dir)

            if not os.path.isdir(word_path):
                continue

            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1

            for file_name in os.listdir(word_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(word_path, file_name)
                    data = np.load(file_path)

                    if len(data) < max_length:
                        pad = np.zeros((max_length - len(data), data.shape[1]))
                        data = np.vstack([data, pad])
                    else:
                        data = data[:max_length]

                    X.append(data)
                    y.append(class_labels[word_dir])

        X = np.array(X)
        y = np.array(y)

        return X, y, class_labels


class SignLanguageDataset(TorchDataset):
    """
    手语数据 PyTorch Dataset

    支持懒加载和可选的数据增强，可直接用于 torch.utils.data.DataLoader。

    使用示例：
    >>> from src.features.augmentation import KeypointAugmenter
    >>> augmenter = KeypointAugmenter(p=0.5)
    >>> dataset = SignLanguageDataset(X, y, augmenter=augmenter)
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, X, y, augmenter=None):
        """
        Args:
            X (np.ndarray): 特征矩阵，形状 (n_samples, ...)
            y (np.ndarray): 标签向量，形状 (n_samples,)
            augmenter (KeypointAugmenter, optional): 数据增强器
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augmenter = augmenter

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # 训练时应用数据增强
        if self.augmenter is not None:
            # 转换为 numpy 进行增强，再转回 tensor
            x_np = x.numpy()
            x_np = self.augmenter(x_np)
            x = torch.tensor(x_np, dtype=torch.float32)

        return x, y

    @property
    def input_shape(self):
        """返回输入形状 (用于模型构建)"""
        return tuple(self.X.shape[1:])

    @property
    def num_classes(self):
        """返回类别数量"""
        return len(torch.unique(self.y))


def create_dataloaders(X, y, batch_size=32, test_size=0.2,
                       augmenter=None, random_state=42):
    """
    便捷函数：从 numpy 数组创建训练/验证 DataLoader

    Args:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 标签向量
        batch_size (int): 批次大小
        test_size (float): 验证集比例
        augmenter: 数据增强器（仅应用于训练集）
        random_state (int): 随机种子

    Returns:
        tuple: (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader as TorchDataLoader

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_dataset = SignLanguageDataset(X_train, y_train, augmenter=augmenter)
    val_dataset = SignLanguageDataset(X_val, y_val, augmenter=None)

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
