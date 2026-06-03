"""
Metrics - 评估指标工具类

本文件提供了分类模型评估指标的计算和可视化功能。

支持的指标：
- 准确率 (Accuracy): 正确预测的样本占总样本的比例
- 精确率 (Precision): 预测为正类的样本中实际为正类的比例
- 召回率 (Recall): 实际为正类的样本中被正确预测的比例
- F1分数 (F1-Score): 精确率和召回率的调和平均数
- 混淆矩阵 (Confusion Matrix): 可视化分类结果

使用示例：
>>> metrics = Metrics()
>>> metrics_dict = metrics.calculate_metrics(y_true, y_pred)
>>> metrics.print_metrics(metrics_dict)
>>> metrics.plot_confusion_matrix(y_true, y_pred, class_names)
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Metrics:
    """
    分类评估指标计算器

    提供多种评估指标的计算方法，并支持混淆矩阵的可视化。
    """

    def __init__(self):
        """初始化指标计算器"""
        pass

    def calculate_metrics(self, y_true, y_pred):
        """
        计算分类评估指标

        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签

        Returns:
            dict: 包含以下指标的字典：
                - accuracy: 准确率
                - precision: 精确率（加权平均）
                - recall: 召回率（加权平均）
                - f1: F1分数（加权平均）
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        绘制混淆矩阵热力图

        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签
            class_names (list, optional): 类别名称列表. Defaults to None.
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        # 设置标签和标题
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # 如果提供了类别名称，设置刻度标签
        if class_names:
            plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
            plt.yticks(np.arange(len(class_names)), class_names, rotation=0)

        plt.title('Confusion Matrix')
        plt.show()

    def print_metrics(self, metrics):
        """
        打印评估指标

        Args:
            metrics (dict): 指标字典
        """
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')
