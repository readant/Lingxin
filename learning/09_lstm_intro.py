"""
第9阶段：深度学习入门 - LSTM时序模型

本脚本帮助您学习LSTM（长短期记忆网络）的基本原理和使用方法。
"""

import numpy as np

def section_1_sequence_intro():
    """9.1 时序数据介绍"""
    print("\n" + "=" * 50)
    print("9.1 时序数据介绍")
    print("=" * 50)

    print("""
什么是时序数据？
---------------
时序数据是按照时间顺序排列的数据，每个数据点都与时间戳相关联。

特点：
-----
1. 数据具有时间顺序
2. 相邻数据点之间存在依赖关系
3. 过去的信息会影响未来的预测

常见时序数据类型：
-----------------
- 股票价格
- 语音信号
- 视频帧序列
- 传感器数据
- 手语动作序列

手语识别中的时序数据：
---------------------
- 输入：连续的视频帧，每帧提取171维特征
- 输出：手势类别
- 特点：手势是一个时间序列，需要考虑时间顺序

为什么需要LSTM？
---------------
传统机器学习模型（如SVM）无法处理时序依赖关系。
LSTM可以记住序列中的长期依赖信息。
""")

def section_2_lstm_principle():
    """9.2 LSTM原理"""
    print("\n" + "=" * 50)
    print("9.2 LSTM原理")
    print("=" * 50)

    print("""
什么是LSTM？
-----------
LSTM（Long Short-Term Memory）是一种特殊的循环神经网络（RNN）。

普通RNN的问题：
-------------
- 梯度消失问题：长序列中，早期信息会丢失
- 无法学习长期依赖关系

LSTM的解决方案：
---------------
通过门控机制控制信息的流动：

1. 遗忘门（Forget Gate）：决定丢弃哪些信息
2. 输入门（Input Gate）：决定哪些新信息加入
3. 输出门（Output Gate）：决定输出什么信息

LSTM结构：
----------
┌─────────────────────────────────────────────────────────┐
│                      LSTM单元                          │
├─────────────────────────────────────────────────────────┤
│                                                       │
│   输入 x_t ──────┐                                    │
│                  │                                    │
│                  ▼                                    │
│   ┌──────────────────────────────┐                    │
│   │     遗忘门 (Forget Gate)     │                    │
│   │   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)            │
│   └─────────────┬────────────────┘                    │
│                 │                                    │
│                 ▼                                    │
│   ┌──────────────────────────────┐                    │
│   │     输入门 (Input Gate)      │                    │
│   │   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)            │
│   │   C_t' = tanh(W_c · [h_{t-1}, x_t] + b_c)       │
│   └─────────────┬────────────────┘                    │
│                 │                                    │
│                 ▼                                    │
│   ┌──────────────────────────────┐                    │
│   │      细胞状态更新            │                    │
│   │   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C_t'              │
│   └─────────────┬────────────────┘                    │
│                 │                                    │
│                 ▼                                    │
│   ┌──────────────────────────────┐                    │
│   │     输出门 (Output Gate)     │                    │
│   │   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)            │
│   │   h_t = o_t ⊙ tanh(C_t)                          │
│   └─────────────┬────────────────┘                    │
│                 │                                    │
│                 ▼                                    │
│              输出 h_t                                 │
│                                                       │
└─────────────────────────────────────────────────────────┘

符号说明：
--------
- σ: Sigmoid函数（输出0-1）
- tanh: 双曲正切函数（输出-1~1）
- ⊙: 元素级乘法
- W: 权重矩阵
- b: 偏置向量
""")

def section_3_pytorch_lstm():
    """9.3 使用PyTorch实现LSTM"""
    print("\n" + "=" * 50)
    print("9.3 使用PyTorch实现LSTM")
    print("=" * 50)

    import torch
    import torch.nn as nn

    print("PyTorch LSTM参数说明:")
    print("""
nn.LSTM(
    input_size: int,      # 输入特征维度
    hidden_size: int,     # 隐藏层维度
    num_layers: int=1,    # LSTM层数
    batch_first: bool=False,  # 输入是否为(batch, seq_len, features)
    bidirectional: bool=False  # 是否双向LSTM
)
""")

    # 创建LSTM示例
    input_size = 171  # 每帧特征维度
    hidden_size = 128  # 隐藏层维度
    num_layers = 2     # 层数

    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=False
    )

    print(f"LSTM结构:\n{lstm}")

    # 模拟输入数据
    batch_size = 32      # 批次大小
    seq_len = 30         # 序列长度（帧数）

    # 输入形状: (batch_size, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    print(f"\n输入形状: {x.shape}")

    # 前向传播
    output, (h_n, c_n) = lstm(x)
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {h_n.shape}")
    print(f"细胞状态形状: {c_n.shape}")

def section_4_lstm_classifier():
    """9.4 项目中的LSTM模型"""
    print("\n" + "=" * 50)
    print("9.4 项目中的LSTM模型")
    print("=" * 50)

    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.models.lstm_model import LSTMModel
        print("[OK] 成功导入项目的 LSTMModel")
        print("""
项目的 LSTMModel 继承自 BaseModel，内置了：
- 统一的训练循环（train_model方法）
- Early Stopping（早停机制）
- 学习率调度器（ReduceLROnPlateau）
- 自动保存最佳模型
- GPU 自动检测支持

模型结构：
  输入(30, 171) → LSTM(2层, 128隐藏单元) → 全连接(128→64) → 输出(num_classes)
""")

        # 创建模型
        input_shape = (30, 171)  # (序列长度, 特征维度)
        num_classes = 10
        model = LSTMModel(input_shape, num_classes)
        print(f"模型结构:\n{model}")

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n总参数数量: {total_params:,}")

        # 测试前向传播
        import torch
        batch_size = 16
        x = torch.randn(batch_size, *input_shape)
        output = model(x)
        print(f"\n输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")

    except ImportError as e:
        print(f"[ERROR] 无法导入 LSTMModel: {e}")
        print("请确保在项目根目录下运行此脚本")
        print("\n以下是独立的LSTM分类器示例：")

        import torch
        import torch.nn as nn

        class SimpleLSTMClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n[-1])

        model = SimpleLSTMClassifier(171, 128, 2, 10)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")

def section_5_training_example():
    """9.5 项目训练流程"""
    print("\n" + "=" * 50)
    print("9.5 项目训练流程")
    print("=" * 50)

    print("""
项目的训练流程已经封装好，使用非常简单：

步骤1: 预处理数据
  python tools/preprocess.py --split-by-person --train-persons L

步骤2: 训练模型
  python tools/train.py --model lstm
  python tools/train.py --model transformer

步骤3: 评估模型
  python tools/evaluate.py  （选择模型类型）

步骤4: 实时推理
  python tools/inference.py --model lstm --checkpoint models/lstm_model.pth

内部训练循环（BaseModel.train_model）：
""")

    print("""
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证 + 早停 + 学习率调度
    val_loss, val_acc = validate(val_loader)
    scheduler.step(val_loss)
    early_stopping(val_loss, model)
""")

    print("""
BaseModel 内置功能：
- Early Stopping: 验证损失不再改善时自动停止
- LR Scheduler: 自动降低学习率
- 最佳模型保存: 自动保存验证集上表现最好的模型
- GPU 支持: 自动检测并使用 CUDA
""")

def main():
    print("=" * 60)
    print("第9阶段：深度学习入门 - LSTM时序模型")
    print("=" * 60)

    section_1_sequence_intro()
    section_2_lstm_principle()
    section_3_pytorch_lstm()
    section_4_lstm_classifier()
    section_5_training_example()

    print("\n" + "=" * 60)
    print("LSTM学习完成！")
    print("下一步：运行 10_data_collection.py 进行项目实战")
    print("=" * 60)

if __name__ == '__main__':
    main()
