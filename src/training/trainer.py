"""
Trainer - 模型训练器

本文件提供了统一的模型训练接口，支持传统机器学习模型和深度学习模型。

功能特性：
- train_classifier: 训练传统机器学习模型（SVM、随机森林、MLP）
- train_deep_learning: 训练深度学习模型（LSTM、Transformer）
- save_classifier: 保存分类器模型
- load_classifier: 加载分类器模型

支持的模型类型：
- SVM (支持向量机): 适合小数据集，效果稳定
- RF (随机森林): 抗噪声能力强，不易过拟合
- MLP (多层感知机): 可学习复杂模式
- LSTM (长短期记忆网络): 适合序列数据
- Transformer: 并行处理能力强，适合大规模数据
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class Trainer:
    """
    模型训练器类

    提供静态方法用于训练不同类型的模型，无需实例化即可使用。
    """

    @staticmethod
    def train_classifier(X, y, model_type='svm', test_size=0.2, **kwargs):
        """
        训练传统机器学习分类器

        Args:
            X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
            y (np.ndarray): 标签向量，形状为(n_samples,)
            model_type (str, optional): 模型类型，可选'svm'、'rf'、'mlp'. Defaults to 'svm'.
            test_size (float, optional): 测试集比例. Defaults to 0.2.
            **kwargs: 其他参数（预留）

        Returns:
            tuple: (模型字典, 测试准确率)
                   模型字典包含: {'model': 训练好的模型, 'scaler': 标准化器, 'type': 模型类型}
        """
        # 延迟导入，避免不必要的依赖加载
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        # 步骤1：划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42  # 设置随机种子保证可复现
        )

        # 步骤2：数据标准化（重要：对SVM和MLP尤为重要）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # 拟合训练数据
        X_test_scaled = scaler.transform(X_test)        # 使用相同的缩放参数转换测试数据

        # 步骤3：定义模型字典
        models = {
            'svm': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        }

        # 步骤4：获取并训练模型
        model = models.get(model_type)
        if not model:
            raise ValueError(f"未知模型类型: {model_type}，可选类型: svm, rf, mlp")

        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 步骤5：预测并计算准确率
        y_pred = model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)

        # 返回模型数据和准确率
        return {'model': model, 'scaler': scaler, 'type': model_type}, accuracy

    @staticmethod
    def train_deep_learning(X, y, model_class, test_size=0.2, epochs=50,
                            batch_size=32, lr=0.001, device='auto',
                            early_stopping_patience=10, save_best_path=None,
                            X_val=None, y_val=None,
                            **kwargs):
        """
        训练深度学习模型

        Args:
            X (np.ndarray): 训练特征矩阵，形状为(n_samples, seq_len, n_features)
            y (np.ndarray): 训练标签向量
            model_class (class): 模型类（如LSTMModel、TransformerModel）
            test_size (float, optional): X_val为空时，从X中划分的比例. Defaults to 0.2.
            epochs (int, optional): 训练轮数
            batch_size (int, optional): 批次大小
            lr (float, optional): 学习率
            device (str, optional): 训练设备
            early_stopping_patience (int, optional): 早停容忍轮数
            save_best_path (str, optional): 最佳模型保存路径
            X_val (np.ndarray, optional): 预划分的验证集特征（独立于训练集的人员）
            y_val (np.ndarray, optional): 预划分的验证集标签
            **kwargs: 其他参数

        Returns:
            tuple: (训练好的模型, 验证集准确率)
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # 转换为PyTorch张量
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train_t = torch.tensor(y, dtype=torch.long)

        # 如果提供了独立验证集，直接使用；否则从训练集自动划分
        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.long)
        else:
            # 自动划分
            X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            X_train = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_t = torch.tensor(y_train_np, dtype=torch.long)
            X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
            y_val_t = torch.tensor(y_val_np, dtype=torch.long)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 获取输入形状和类别数
        input_size = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(np.concatenate([y, y_val if y_val is not None else []])))

        # 创建并训练模型
        model = model_class(input_size, num_classes)
        model.train_model(
            train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
            early_stopping_patience=early_stopping_patience,
            save_best_path=save_best_path
        )

        # 在验证集上评估
        model.eval()
        with torch.no_grad():
            X_val_t = X_val_t.to(model.device)
            y_val_t = y_val_t.to(model.device)
            outputs = model(X_val_t)
            _, y_pred = torch.max(outputs, 1)
            accuracy = (y_pred == y_val_t).float().mean().item()

        return model, accuracy

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        在独立测试集上评估模型（用于预划分场景）

        Args:
            model: 训练好的深度学习模型
            X_test (np.ndarray): 测试集特征
            y_test (np.ndarray): 测试集标签

        Returns:
            float: 测试准确率
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.long)

        test_dataset = TensorDataset(X_t, y_t)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total

    @staticmethod
    def save_classifier(model_data, path):
        """
        保存分类器模型（包含模型和标准化器）

        Args:
            model_data (dict): 包含'model'、'scaler'、'type'的字典
            path (str): 保存路径（建议使用.pkl扩展名）
        """
        import joblib
        joblib.dump(model_data, path)

    @staticmethod
    def load_classifier(path):
        """
        加载分类器模型

        Args:
            path (str): 模型文件路径

        Returns:
            dict: 包含'model'、'scaler'、'type'的字典
        """
        import joblib
        return joblib.load(path)
