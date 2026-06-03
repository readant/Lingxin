"""
Visualization - 可视化工具类

本文件提供了数据可视化功能，包括：
- 手部关键点可视化
- 模型训练曲线可视化（准确率、损失）

使用示例：
>>> viz = Visualization()
>>> viz.plot_landmarks(hand_landmarks)
>>> viz.plot_accuracy(history)
>>> viz.plot_loss(history)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union

from src.constants import HAND_CONNECTIONS


class Visualization:
    """
    可视化工具类

    提供手部关键点绘制和训练曲线绘制功能。
    支持 PyTorch 训练历史记录（dict 格式）。
    """

    # 手部骨架连接定义（从 src.constants 导入，此处保留引用以兼容旧代码）
    HAND_CONNECTIONS = HAND_CONNECTIONS

    def __init__(self):
        """初始化可视化工具"""
        pass

    def plot_landmarks(self, landmarks, ax=None):
        """
        绘制手部关键点骨架

        Args:
            landmarks (np.ndarray): 手部关键点数组，形状为(21, 3)或(n_hands, 21, 3)
            ax (matplotlib.axes.Axes, optional): 绘图轴对象. Defaults to None.

        Returns:
            matplotlib.axes.Axes: 绘图轴对象
        """
        if ax is None:
            fig, ax = plt.subplots()

        # 处理单只手或多只手的情况
        if landmarks.ndim == 2:
            landmarks = [landmarks]

        # 绘制骨架连接和关键点
        for hand_landmarks in landmarks:
            for connection in self.HAND_CONNECTIONS:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'r-')

            ax.scatter(hand_landmarks[:, 0], hand_landmarks[:, 1], s=20, c='b')

        # 设置坐标轴属性
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 翻转Y轴以匹配图像坐标系

        return ax

    def plot_accuracy(self, history: Dict[str, List[float]],
                      title: str = 'Model Accuracy',
                      figsize: tuple = (10, 6)):
        """
        绘制训练准确率曲线

        支持两种输入格式：
        1. dict with 'val_accuracy' (PyTorch BaseModel.train_model 返回的格式)
        2. dict with 'accuracy' and 'val_accuracy' keys

        Args:
            history (dict): 训练历史记录字典，期望包含以下键：
                - 'val_accuracy': 验证准确率列表
                - 'accuracy': 训练准确率列表（可选）
            title (str, optional): 图表标题. Defaults to 'Model Accuracy'.
            figsize (tuple, optional): 图表尺寸. Defaults to (10, 6).
        """
        plt.figure(figsize=figsize)

        if 'accuracy' in history:
            plt.plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Validation Accuracy')

        if not ('accuracy' in history or 'val_accuracy' in history):
            print("警告：history 中未找到 'accuracy' 或 'val_accuracy' 键")
            return

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_loss(self, history: Dict[str, List[float]],
                  title: str = 'Model Loss',
                  figsize: tuple = (10, 6)):
        """
        绘制训练损失曲线

        支持两种输入格式：
        1. dict with 'train_loss' and 'val_loss' (PyTorch BaseModel.train_model 返回的格式)
        2. dict with 'loss' and 'val_loss' keys

        Args:
            history (dict): 训练历史记录字典，期望包含以下键：
                - 'train_loss': 训练损失列表
                - 'val_loss': 验证损失列表（可选）
                - 'loss': 训练损失列表（兼容旧格式，可选）
            title (str, optional): 图表标题. Defaults to 'Model Loss'.
            figsize (tuple, optional): 图表尺寸. Defaults to (10, 6).
        """
        plt.figure(figsize=figsize)

        train_loss = history.get('train_loss', history.get('loss'))
        val_loss = history.get('val_loss')

        if train_loss:
            plt.plot(train_loss, label='Training Loss')
        if val_loss:
            plt.plot(val_loss, label='Validation Loss')

        if not (train_loss or val_loss):
            print("警告：history 中未找到 'train_loss'、'loss' 或 'val_loss' 键")
            return

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_training_curves(self, history: Dict[str, List[float]],
                             figsize: tuple = (12, 5)):
        """
        同时绘制损失和准确率曲线（并排子图）

        Args:
            history (dict): 训练历史记录字典
            figsize (tuple, optional): 图表尺寸. Defaults to (12, 5).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 损失曲线
        train_loss = history.get('train_loss', history.get('loss', []))
        val_loss = history.get('val_loss', [])
        if train_loss:
            ax1.plot(train_loss, label='Training Loss')
        if val_loss:
            ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        train_acc = history.get('accuracy', [])
        val_acc = history.get('val_accuracy', [])
        if train_acc:
            ax2.plot(train_acc, label='Training Accuracy')
        if val_acc:
            ax2.plot(val_acc, label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
