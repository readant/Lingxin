"""
LSTMModel - 长短期记忆网络模型

本文件实现了基于LSTM的手语识别模型。

模型架构：
输入 → LSTM层(2层) → 取最后时刻输出 → 全连接层(128→64) → 输出层(64→num_classes)

LSTM（Long Short-Term Memory）是一种特殊的循环神经网络，
通过门控机制能够有效捕捉序列数据中的长期依赖关系。

适用场景：
- 时序数据建模
- 手语动作序列识别
- 需要捕捉时间依赖的任务
"""

import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM手语识别模型

    继承自BaseModel，实现了forward方法定义模型结构。
    训练流程由BaseModel的train_model方法统一管理。
    """

    def __init__(self, input_shape, num_classes):
        """
        初始化LSTM模型

        Args:
            input_shape (tuple): 输入形状，格式为(seq_len, feature_dim)
            num_classes (int): 输出类别数量
        """
        super().__init__()

        # 从输入形状中提取参数
        self.input_size = input_shape[1]  # 特征维度
        self.hidden_size = 128           # LSTM隐藏层大小
        self.num_layers = 2              # LSTM层数

        # 定义LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,    # 输入特征维度
            hidden_size=self.hidden_size,  # 隐藏层大小
            num_layers=self.num_layers,    # 层数
            batch_first=True,              # 输入格式为(batch, seq, feature)
            dropout=0.2                    # Dropout比例，防止过拟合
        )

        # 定义全连接层
        self.fc1 = nn.Linear(self.hidden_size, 64)  # 隐藏层到64维
        self.fc2 = nn.Linear(64, num_classes)       # 64维到类别数
        self.relu = nn.ReLU()                        # ReLU激活函数

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_len, feature_dim)

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, num_classes)
        """
        # 初始化隐藏状态和细胞状态
        # 形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        # out: (batch_size, seq_len, hidden_size)
        # _: (h_n, c_n) 最后时刻的隐藏状态和细胞状态（未使用）
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出（用于分类任务）
        # out[:, -1, :]: (batch_size, hidden_size)
        out = out[:, -1, :]

        # 全连接层处理
        out = self.relu(self.fc1(out))  # (batch_size, 64)
        out = self.fc2(out)              # (batch_size, num_classes)

        return out
