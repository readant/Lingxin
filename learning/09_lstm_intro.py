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
    """9.4 LSTM分类器实现"""
    print("\n" + "=" * 50)
    print("9.4 LSTM分类器实现")
    print("=" * 50)
    
    import torch
    import torch.nn as nn
    
    class LSTMClassifier(nn.Module):
        """用于手语识别的LSTM分类器"""
        
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(LSTMClassifier, self).__init__()
            
            # LSTM层
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            
            # 全连接层
            self.fc = nn.Linear(hidden_size, num_classes)
            
            # 激活函数
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            """
            参数:
                x: 输入张量，形状为(batch_size, seq_len, input_size)
            
            返回:
                预测概率，形状为(batch_size, num_classes)
            """
            # LSTM前向传播
            _, (h_n, _) = self.lstm(x)
            
            # 取最后一层的隐藏状态
            last_hidden = h_n[-1]  # (batch_size, hidden_size)
            
            # 全连接层
            logits = self.fc(last_hidden)
            
            # Softmax激活
            probabilities = self.softmax(logits)
            
            return probabilities
    
    # 创建模型
    input_size = 171    # 171维特征
    hidden_size = 128   # 隐藏层维度
    num_layers = 2      # LSTM层数
    num_classes = 10    # 手势类别数量
    
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    print(f"模型结构:\n{model}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数量: {total_params:,}")
    
    # 测试前向传播
    batch_size = 16
    seq_len = 30
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例:\n{output[0].detach().numpy().round(4)}")

def section_5_training_example():
    """9.5 训练示例"""
    print("\n" + "=" * 50)
    print("9.5 训练示例")
    print("=" * 50)
    
    import torch
    import torch.nn as nn
    
    # 超参数
    input_size = 171
    hidden_size = 128
    num_layers = 2
    num_classes = 5
    batch_size = 32
    seq_len = 30
    epochs = 10
    
    # 创建模型
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("训练循环流程:")
    print("""
for epoch in range(epochs):
    model.train()
    
    # 前向传播
    outputs = model(inputs)
    
    # 计算损失
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs)
        accuracy = calculate_accuracy(predictions, test_labels)
""")
    
    # 模拟训练数据
    print("\n模拟训练:")
    for epoch in range(epochs):
        # 生成模拟数据
        inputs = torch.randn(batch_size, seq_len, input_size)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    print("\n训练完成！")

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