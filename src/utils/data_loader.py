"""
DataLoader - 数据加载器

本文件实现了手语数据的加载功能，支持两种数据格式：
1. 特征数据：用于传统机器学习模型（SVM、RF、MLP）
2. 序列数据：用于深度学习模型（LSTM、Transformer）

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
        X = []  # 特征列表
        y = []  # 标签列表
        class_labels = {}  # 类别标签映射
        label_idx = 0      # 类别索引计数器
        
        # 遍历数据目录中的所有子目录（每个子目录对应一个词汇）
        for word_dir in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word_dir)
            
            # 跳过非目录文件
            if not os.path.isdir(word_path):
                continue
            
            # 如果是新类别，分配新的类别索引
            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1
            
            # 遍历该类别下的所有.npy文件
            for file_name in os.listdir(word_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(word_path, file_name)
                    # 加载特征数据
                    data = np.load(file_path)
                    X.append(data)
                    y.append(class_labels[word_dir])
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        
        return X, y, class_labels
    
    def load_sequence_data(self, data_dir, max_length=30):
        """
        加载序列数据（用于深度学习模型）
        
        序列数据需要统一长度，通过padding或截断实现。
        
        Args:
            data_dir (str): 数据目录路径，包含各个词汇的子目录
            max_length (int, optional): 最大序列长度. Defaults to 30.
            
        Returns:
            tuple: (X, y, class_labels)
                   X: 序列特征矩阵，形状为(n_samples, max_length, n_features)
                   y: 标签向量，形状为(n_samples,)
                   class_labels: 类别标签字典，{类别名: 类别索引}
        """
        X = []  # 序列特征列表
        y = []  # 标签列表
        class_labels = {}  # 类别标签映射
        label_idx = 0      # 类别索引计数器
        
        # 遍历数据目录中的所有子目录
        for word_dir in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word_dir)
            
            # 跳过非目录文件
            if not os.path.isdir(word_path):
                continue
            
            # 如果是新类别，分配新的类别索引
            if word_dir not in class_labels:
                class_labels[word_dir] = label_idx
                label_idx += 1
            
            # 遍历该类别下的所有.npy文件
            for file_name in os.listdir(word_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(word_path, file_name)
                    # 加载序列数据
                    data = np.load(file_path)
                    
                    # 统一序列长度
                    if len(data) < max_length:
                        # 如果序列长度不足，进行padding（用0填充）
                        pad = np.zeros((max_length - len(data), data.shape[1]))
                        data = np.vstack([data, pad])
                    else:
                        # 如果序列过长，进行截断
                        data = data[:max_length]
                    
                    X.append(data)
                    y.append(class_labels[word_dir])
        
        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        
        return X, y, class_labels