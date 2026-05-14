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


class Visualization:
    """
    可视化工具类

    提供手部关键点绘制和训练曲线绘制功能。
    """

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
        # 如果没有提供轴对象，创建新图
        if ax is None:
            fig, ax = plt.subplots()

        # 手部骨架连接定义（MediaPipe标准）
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]

        # 处理单只手或多只手的情况
        if landmarks.ndim == 2:
            landmarks = [landmarks]

        # 绘制骨架连接和关键点
        for hand_landmarks in landmarks:
            for connection in connections:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'r-')

            ax.scatter(hand_landmarks[:, 0], hand_landmarks[:, 1], s=20, c='b')

        # 设置坐标轴属性
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 翻转Y轴以匹配图像坐标系

        return ax

    def plot_accuracy(self, history):
        """
        绘制训练准确率曲线

        Args:
            history (keras.callbacks.History): Keras训练历史对象
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')

        # 如果有验证准确率，也绘制出来
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        plt.show()

    def plot_loss(self, history):
        """
        绘制训练损失曲线

        Args:
            history (keras.callbacks.History): Keras训练历史对象
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')

        # 如果有验证损失，也绘制出来
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.show()