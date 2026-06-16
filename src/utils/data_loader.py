"""
DataLoader - 数据加载器

本文件实现了手语数据的加载功能，支持两种数据格式：
1. 特征数据：用于传统机器学习模型（SVM、RF、MLP）
2. 序列数据：用于深度学习模型（LSTM、Transformer）

同时还提供标准 PyTorch Dataset 类，可直接用于 torch.utils.data.DataLoader。

数据目录结构要求（按人员命名）：
data/raw/collected/
└── {word}/
    ├── {person_id}_{index:03d}.npy       # F_001.npy, L_001.npy
    └── {person_id}_{index:03d}_meta.json  # F_001_meta.json (可选)

文件命名规范：
  {person_id}_{序号}.npy  → person_id 从第一个 '_' 前提取
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from collections import defaultdict


def parse_person_id(filename):
    """
    从文件名解析人员ID

    支持格式：
      F_001.npy       → 'F'
      L_010.npy       → 'L'
      user001_005.npy → 'user001'

    Args:
        filename (str): 文件名

    Returns:
        str: 人员ID，解析失败返回 'unknown'
    """
    name = os.path.splitext(filename)[0]           # 去掉扩展名 → 'F_001'
    # 去掉 _meta 后缀（如果有）
    if name.endswith('_meta'):
        return None
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0]                             # 第一个 '_' 之前 = person_id
    return 'unknown'


class DataLoader:
    """
    手语数据加载器

    负责从磁盘加载预处理后的手语数据，支持特征数据和序列数据两种格式。
    可从文件名自动解析人员ID，支持按人员划分数据集。
    """

    def __init__(self):
        """初始化数据加载器"""
        pass

    def load_data(self, data_dir, return_persons=False):
        """
        加载特征数据（用于传统机器学习模型）

        Args:
            data_dir (str): 数据目录路径，包含各个词汇的子目录
            return_persons (bool): 是否同时返回人员ID列表

        Returns:
            tuple: (X, y, class_labels) 或 (X, y, class_labels, person_ids)
        """
        X = []
        y = []
        class_labels = {}
        person_ids = []
        label_idx = 0

        for word_dir in sorted(os.listdir(data_dir)):
            word_path = os.path.join(data_dir, word_dir)

            if not os.path.isdir(word_path):
                continue

            # 只处理有.npy文件的目录
            npy_files = [f for f in os.listdir(word_path)
                        if f.endswith('.npy') and not f.endswith('_meta.npy')]
            if not npy_files:
                continue

            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1

            for file_name in sorted(npy_files):
                file_path = os.path.join(word_path, file_name)
                data = np.load(file_path)
                # 如果是序列数据(n_frames, n_features)，提取中间帧作为单帧特征
                if data.ndim == 2:
                    mid_idx = len(data) // 2
                    data = data[mid_idx]
                X.append(data)
                y.append(class_labels[word_dir])
                if return_persons:
                    person_ids.append(parse_person_id(file_name))

        X = np.array(X)
        y = np.array(y)

        if return_persons:
            return X, y, class_labels, np.array(person_ids)
        return X, y, class_labels

    def load_sequence_data(self, data_dir, max_length=30, return_persons=False):
        """
        加载序列数据（用于深度学习模型）

        Args:
            data_dir (str): 数据目录路径
            max_length (int, optional): 最大序列长度. Defaults to 30.
            return_persons (bool): 是否同时返回人员ID列表

        Returns:
            tuple: (X, y, class_labels) 或 (X, y, class_labels, person_ids)
        """
        X = []
        y = []
        class_labels = {}
        person_ids = []
        label_idx = 0

        for word_dir in sorted(os.listdir(data_dir)):
            word_path = os.path.join(data_dir, word_dir)

            if not os.path.isdir(word_path):
                continue

            # 只处理有.npy文件的目录
            npy_files = [f for f in os.listdir(word_path)
                        if f.endswith('.npy') and not f.endswith('_meta.npy')]
            if not npy_files:
                continue

            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1

            for file_name in sorted(npy_files):
                file_path = os.path.join(word_path, file_name)
                data = np.load(file_path)

                if len(data) < max_length:
                    pad = np.zeros((max_length - len(data), data.shape[1]))
                    data = np.vstack([data, pad])
                else:
                    data = data[:max_length]

                X.append(data)
                y.append(class_labels[word_dir])
                if return_persons:
                    person_ids.append(parse_person_id(file_name))

        X = np.array(X)
        y = np.array(y)

        if return_persons:
            return X, y, class_labels, np.array(person_ids)
        return X, y, class_labels

    def split_by_person(self, X, y, person_ids, train_persons, val_persons=None, test_persons=None):
        """
        按人员划分数据集

        同一人员的数据保证不会出现在多个集合中。

        Args:
            X (np.ndarray): 特征矩阵
            y (np.ndarray): 标签向量
            person_ids (np.ndarray): 人员ID数组
            train_persons (list): 训练集人员ID列表
            val_persons (list, optional): 验证集人员ID列表
            test_persons (list, optional): 测试集人员ID列表

        Returns:
            dict: {
                'train': (X_train, y_train),
                'val': (X_val, y_val) or None,
                'test': (X_test, y_test) or None,
            }
        """
        result = {}

        for split_name, persons in [
            ('train', train_persons),
            ('val', val_persons),
            ('test', test_persons),
        ]:
            if not persons:
                result[split_name] = None
                continue

            mask = np.isin(person_ids, persons)
            if not mask.any():
                result[split_name] = None
                continue

            result[split_name] = (X[mask].copy(), y[mask].copy())

        return result

    def auto_split_persons(self, person_ids, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        自动按人员随机划分 train/val/test

        Args:
            person_ids (np.ndarray): 所有样本的人员ID
            val_ratio (float): 验证集人员比例
            test_ratio (float): 测试集人员比例
            random_state (int): 随机种子

        Returns:
            dict: {'train': [...], 'val': [...], 'test': [...]}
        """
        unique_persons = sorted(set(person_ids))
        n_total = len(unique_persons)

        if n_total < 3:
            # 人员太少，全部放训练集
            return {'train': unique_persons, 'val': [], 'test': []}

        np.random.seed(random_state)
        indices = np.random.permutation(n_total)

        n_test = max(1, int(n_total * test_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return {
            'train': [unique_persons[i] for i in train_idx],
            'val': [unique_persons[i] for i in val_idx] if n_val > 0 else [],
            'test': [unique_persons[i] for i in test_idx] if n_test > 0 else [],
        }


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
