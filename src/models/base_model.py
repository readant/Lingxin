"""
BaseModel - 深度学习模型抽象基类（模板方法模式）

本文件定义了深度学习模型的抽象基类，通过模板方法模式统一训练流程。
所有深度学习模型（如LSTM、Transformer）都应继承此类，并实现forward方法。

设计模式：模板方法模式
- 定义算法骨架（训练流程）
- 将具体步骤（模型结构）延迟到子类实现

功能特性：
- 统一的训练循环
- Early Stopping（早停机制）
- 学习率调度器（ReduceLROnPlateau）
- 自动保存最佳模型
- GPU 自动检测支持
- 模型保存/加载
- 预测接口
"""

import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils.logger import get_logger


class EarlyStopping:
    """
    早停机制

    监控验证损失，当损失在 patience 轮内未改善时停止训练。
    同时保存最佳模型状态。

    Attributes:
        patience: 容忍轮数
        min_delta: 最小改善阈值
        best_score: 最佳分数（负数，越大越好）
        counter: 未改善计数器
        best_model_state: 最佳模型的状态字典
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: 容忍轮数，超过此轮数未改善则停止
            min_delta: 最小改善阈值，低于此值视为未改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应早停

        Args:
            val_loss: 当前验证损失
            model: 当前模型（用于保存最佳状态）

        Returns:
            bool: True 表示应继续训练，False 表示应停止
        """
        score = -val_loss  # 损失越低越好，取负数使比较方向一致

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)

        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(self, model: nn.Module):
        """保存最佳模型状态的深拷贝"""
        self.best_model_state = copy.deepcopy(model.state_dict())

    def load_best(self, model: nn.Module):
        """将最佳模型状态加载回模型"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class BaseModel(ABC, nn.Module):
    """
    深度学习模型抽象基类

    所有深度学习模型必须继承此类，并实现抽象方法forward()。
    基类提供了统一的训练流程、验证、预测和保存加载功能。
    """

    def __init__(self):
        """初始化基类，调用父类nn.Module的构造函数"""
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device('cpu')

    def to_device(self, device: str = 'auto'):
        """
        将模型移动到指定设备

        Args:
            device: 'cpu', 'cuda', 或 'auto'（自动检测）
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.to(self.device)
        self.logger.info(f'模型已移动到: {self.device}')
        return self

    @abstractmethod
    def forward(self, x):
        """
        前向传播方法（抽象方法，必须在子类中实现）

        Args:
            x (torch.Tensor): 输入张量，形状取决于具体模型

        Returns:
            torch.Tensor: 输出张量，通常为形状(batch_size, num_classes)
        """
        pass

    def train_model(self, train_loader, val_loader=None, epochs=50, lr=0.001,
                    device='auto', early_stopping_patience=10,
                    lr_scheduler_patience=5, lr_scheduler_factor=0.5,
                    save_best_path=None):
        """
        模板方法：统一的训练流程（支持 Early Stopping 和 LR Scheduler）

        Args:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            epochs (int): 训练轮数. Defaults to 50.
            lr (float): 学习率. Defaults to 0.001.
            device (str): 训练设备. Defaults to 'auto'.
            early_stopping_patience (int): 早停容忍轮数. Defaults to 10.
            lr_scheduler_patience (int): LR调度器容忍轮数. Defaults to 5.
            lr_scheduler_factor (float): LR调度器衰减因子. Defaults to 0.5.
            save_best_path (str, optional): 最佳模型保存路径

        Returns:
            dict: 训练历史记录
        """
        # 设备设置
        self.to_device(device)

        # 获取损失函数和优化器
        criterion = self._get_criterion()
        optimizer = self._get_optimizer(lr)

        # 学习率调度器（当验证损失停止改善时降低学习率）
        scheduler = None
        if val_loader is not None and lr_scheduler_patience > 0:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_scheduler_factor,
                patience=lr_scheduler_patience, verbose=False
            )

        # 早停机制
        early_stopping = None
        if val_loader is not None and early_stopping_patience > 0:
            early_stopping = EarlyStopping(patience=early_stopping_patience)

        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }

        # 训练主循环
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                # 数据移动到设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            # 计算平均损失
            epoch_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)

            # 验证
            if val_loader:
                val_loss, val_accuracy = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                # 更新学习率调度器
                if scheduler is not None:
                    scheduler.step(val_loss)

                # 检查早停
                if early_stopping is not None:
                    should_stop = early_stopping(val_loss, self)
                    if should_stop:
                        self.logger.info(
                            f'Early stopping 触发于 Epoch {epoch+1}，'
                            f'最佳 Val Loss: {-early_stopping.best_score:.4f}'
                        )
                        break

                self.logger.info(
                    f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
                )
            else:
                self.logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)

        # 恢复最佳模型状态
        if early_stopping is not None and early_stopping.best_model_state is not None:
            early_stopping.load_best(self)
            history['best_epoch'] = epoch - early_stopping.counter
            self.logger.info(f'已恢复最佳模型 (Epoch {history["best_epoch"]})')

            # 保存最佳模型
            if save_best_path:
                self.save(save_best_path)
                self.logger.info(f'最佳模型已保存至: {save_best_path}')

        return history

    def _get_criterion(self):
        """
        获取损失函数（钩子方法，可被子类重写）

        Returns:
            nn.Module: 损失函数，默认为CrossEntropyLoss
        """
        return nn.CrossEntropyLoss()

    def _get_optimizer(self, lr):
        """
        获取优化器（钩子方法，可被子类重写）

        Args:
            lr (float): 学习率

        Returns:
            torch.optim.Optimizer: 优化器，默认为Adam
        """
        return optim.Adam(self.parameters(), lr=lr)

    def _validate(self, val_loader, criterion):
        """
        验证模型性能（内部方法）

        Args:
            val_loader (DataLoader): 验证数据加载器
            criterion (nn.Module): 损失函数

        Returns:
            tuple: (平均验证损失, 验证准确率)
        """
        self.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        return val_loss, val_accuracy

    def predict(self, x):
        """
        预测方法（用于推理阶段）

        Args:
            x (torch.Tensor or np.ndarray): 输入数据

        Returns:
            torch.Tensor: 预测标签（类别索引）
        """
        self.eval()

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def save(self, path):
        """
        保存模型参数

        Args:
            path (str): 保存路径（通常以.pt或.pth为扩展名）
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        加载模型参数

        Args:
            path (str): 模型参数文件路径
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
