"""
EvaluateRunner - 模型评估运行器

本文件提供了统一的模型评估接口，支持评估多种模型类型的性能。

支持的模型类型：
- 传统机器学习：SVM、随机森林、MLP
- 深度学习：LSTM、Transformer

评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)

使用方法：
python tools/evaluate.py
然后按照提示输入模型类型。
"""

import numpy as np
import os
from src.training.trainer import Trainer


class EvaluateRunner:
    """
    模型评估运行器

    负责加载数据、创建模型、运行评估并输出结果。
    使用字典映射消除 if-elif 分支，提高可维护性。
    """

    # 模型配置映射
    # 键: 模型类型名称
    # 值: 配置字典
    MODEL_CONFIG = {
        'svm': {
            'data_type': 'classifier',      # 数据类型
            'input_shape': None,            # 传统ML不需要指定形状
            'epochs': None,                  # 传统ML不需要训练轮数
        },
        'rf': {
            'data_type': 'classifier',
            'input_shape': None,
            'epochs': None,
        },
        'mlp': {
            'data_type': 'classifier',
            'input_shape': None,
            'epochs': None,
        },
        'lstm': {
            'data_type': 'sequence',        # 序列数据
            'input_shape': (30, 71),         # (序列长度, 特征维度)
            'epochs': 50,
        },
        'transformer': {
            'data_type': 'sequence',
            'input_shape': (30, 71),
            'epochs': 50,
        }
    }

    # 数据文件配置
    DATA_FILES = {
        'classifier': {'X': 'X.npy', 'y': 'y.npy'},
        'sequence': {'X': 'X_sequence.npy', 'y': 'y_sequence.npy'}
    }

    def __init__(self):
        """初始化评估运行器"""
        self.metrics_calculator = MetricsCalculator()

    def run(self, data_dir, model_type='svm', model_path=None):
        """
        执行模型评估

        Args:
            data_dir (str): 数据目录路径
            model_type (str, optional): 模型类型. Defaults to 'svm'.
            model_path (str, optional): 模型文件路径（预留）. Defaults to None.
        """
        # 验证模型类型
        if model_type not in self.MODEL_CONFIG:
            print(f'未知模型类型: {model_type}，可选类型: {", ".join(self.MODEL_CONFIG.keys())}')
            return

        config = self.MODEL_CONFIG[model_type]
        data_type = config['data_type']
        data_files = self.DATA_FILES[data_type]

        # 加载数据
        print(f'正在加载 {data_type} 数据...')
        X = np.load(os.path.join(data_dir, data_files['X']))
        y = np.load(os.path.join(data_dir, data_files['y']))
        print(f'数据加载完成！样本数: {X.shape[0]}, 特征形状: {X.shape[1:]}')

        # 根据数据类型选择评估方法
        if data_type == 'classifier':
            self._evaluate_classifier(X, y, model_type)
        else:
            self._evaluate_deep_learning(X, y, model_type, config)

    def _evaluate_classifier(self, X, y, model_type):
        """
        评估传统机器学习分类器

        Args:
            X (np.ndarray): 特征矩阵
            y (np.ndarray): 标签向量
            model_type (str): 模型类型
        """
        print(f'正在评估 {model_type} 分类器...')

        # 训练模型并获取预测结果
        model_data, accuracy = Trainer.train_classifier(X, y, model_type)
        model = model_data['model']
        scaler = model_data['scaler']

        # 在测试集上预测
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        # 计算评估指标
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred)
        print(f'{model_type} 分类器评估结果:')
        self._print_metrics(metrics)

    def _evaluate_deep_learning(self, X, y, model_type, config):
        """
        评估深度学习模型

        Args:
            X (np.ndarray): 序列特征矩阵
            y (np.ndarray): 标签向量
            model_type (str): 模型类型
            config (dict): 模型配置
        """
        print(f'正在评估 {model_type} 模型...')

        # 获取模型类
        model_class = self._get_deep_learning_model_class(model_type)

        # 训练模型并获取预测结果
        model, accuracy = Trainer.train_deep_learning(
            X, y, model_class,
            epochs=config['epochs']
        )

        # 在测试集上预测
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.eval()
        with np.no_errstate():
            X_test_tensor = np.array(X_test, dtype=np.float32)
            y_pred = model.predict(X_test_tensor)
            y_pred = np.argmax(y_pred, axis=1)

        # 计算评估指标
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred)
        print(f'{model_type} 模型评估结果:')
        self._print_metrics(metrics)

    def _get_deep_learning_model_class(self, model_type):
        """
        获取深度学习模型类

        Args:
            model_type (str): 模型类型

        Returns:
            class: 模型类
        """
        model_classes = {
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        return model_classes.get(model_type)

    def _print_metrics(self, metrics):
        """
        打印评估指标

        Args:
            metrics (dict): 指标字典
        """
        print('-' * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f'  {key}: {value:.4f}')
            else:
                print(f'  {key}: {value}')
        print('-' * 40)


class MetricsCalculator:
    """
    评估指标计算器

    提供准确率、精确率、召回率、F1分数等分类指标的計算功能。
    """

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        计算分类评估指标

        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签

        Returns:
            dict: 指标字典，包含 accuracy, precision, recall, f1
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # 去除单维度（如果存在）
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }


if __name__ == '__main__':
    runner = EvaluateRunner()
    data_dir = 'data/processed/csl_isolated'
    model_type = input('请选择模型类型 (svm/rf/mlp/lstm/transformer): ')
    runner.run(data_dir, model_type)