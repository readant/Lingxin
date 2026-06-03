"""
TransformerModel - Transformer编码器模型

本文件实现了基于Transformer的手语识别模型。

模型架构：
输入 → 线性嵌入 → 位置编码 → Transformer编码器(2层) → 全局平均池化 → 全连接层 → 输出

Transformer是一种基于自注意力机制的序列建模架构，
能够并行处理序列中的所有位置，同时捕捉长距离依赖。

适用场景：
- 大规模序列数据
- 需要并行处理的任务
- 复杂的依赖关系建模
"""

import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class TransformerModel(BaseModel):
    """
    Transformer手语识别模型

    继承自BaseModel，实现了forward方法定义模型结构。
    训练流程由BaseModel的train_model方法统一管理。
    """

    def __init__(self, input_shape, num_classes, d_model=64, num_heads=2, dff=128, num_layers=2, dropout_rate=0.1):
        """
        初始化Transformer模型

        Args:
            input_shape (tuple): 输入形状，格式为(seq_len, feature_dim)
            num_classes (int): 输出类别数量
            d_model (int, optional): 模型维度（嵌入维度）. Defaults to 64.
            num_heads (int, optional): 注意力头数量. Defaults to 2.
            dff (int, optional): 前馈网络隐藏层维度. Defaults to 128.
            num_layers (int, optional): Transformer编码器层数. Defaults to 2.
            dropout_rate (float, optional): Dropout比例. Defaults to 0.1.
        """
        super().__init__()

        # 从输入形状中提取参数
        self.input_size = input_shape[1]  # 特征维度
        self.d_model = d_model            # 模型维度
        self.num_layers = num_layers      # Transformer层数

        # 线性嵌入层：将输入特征映射到d_model维度
        self.embedding = nn.Linear(self.input_size, d_model)

        # 位置编码：为序列添加位置信息
        self.pos_encoding = self._generate_positional_encoding(input_shape[0], d_model)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              # 模型维度
            nhead=num_heads,              # 注意力头数量
            dim_feedforward=dff,          # 前馈网络维度
            dropout=dropout_rate,         # Dropout比例
            batch_first=True              # 输入格式为(batch, seq, feature)
        )

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 分类头
        self.fc1 = nn.Linear(d_model, 64)      # 隐藏层
        self.fc2 = nn.Linear(64, num_classes)  # 输出层
        self.relu = nn.ReLU()                   # ReLU激活函数
        self.dropout = nn.Dropout(dropout_rate) # Dropout层

    def _generate_positional_encoding(self, max_seq_len, d_model):
        """
        生成正弦余弦位置编码

        位置编码公式：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            max_seq_len (int): 最大序列长度
            d_model (int): 模型维度

        Returns:
            torch.Tensor: 位置编码，形状为(1, max_seq_len, d_model)
        """
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)

        # 位置索引
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 偶数位置使用正弦
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数位置使用余弦
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加batch维度
        pe = pe.unsqueeze(0)

        return pe

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, seq_len, feature_dim)

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, num_classes)
        """
        # 线性嵌入：(batch, seq_len, input_size) → (batch, seq_len, d_model)
        x = self.embedding(x)

        # 按比例缩放嵌入（Transformer论文中的标准做法）
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # 添加位置编码（只取对应序列长度的部分）
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)

        # Transformer编码：(batch, seq_len, d_model) → (batch, seq_len, d_model)
        out = self.transformer_encoder(x)

        # 全局平均池化：(batch, seq_len, d_model) → (batch, d_model)
        out = out.mean(dim=1)

        # 分类头处理
        out = self.relu(self.fc1(out))  # (batch, 64)
        out = self.dropout(out)          # Dropout
        out = self.fc2(out)              # (batch, num_classes)

        return out
