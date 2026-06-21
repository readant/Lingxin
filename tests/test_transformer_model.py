"""测试 src/models/transformer_model.py — TransformerModel"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from src.models.transformer_model import TransformerModel


class TestTransformerModel:
    """测试Transformer模型"""

    @pytest.fixture
    def model(self):
        """创建Transformer模型实例"""
        input_shape = (30, 71)  # seq_len=30, feature_dim=71
        num_classes = 5
        return TransformerModel(input_shape, num_classes)

    def test_init(self, model):
        """模型初始化应正确设置参数"""
        assert model.input_size == 71
        assert model.d_model == 64

    def test_forward_shape(self, model):
        """前向传播输出形状应正确"""
        batch_size = 4
        seq_len = 30
        n_features = 71
        
        x = torch.randn(batch_size, seq_len, n_features)
        output = model(x)
        
        assert output.shape == (batch_size, 5)  # num_classes=5

    def test_forward_different_batch(self, model):
        """不同batch_size应正常工作"""
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 30, 71)
            output = model(x)
            assert output.shape[0] == batch_size

    def test_forward_different_seq_len(self, model):
        """不同序列长度应正常工作（不超过初始化时的最大长度）"""
        for seq_len in [10, 20, 30]:
            x = torch.randn(2, seq_len, 71)
            output = model(x)
            assert output.shape == (2, 5)

    def test_custom_params(self):
        """自定义参数应正确生效"""
        model = TransformerModel(
            input_shape=(30, 71),
            num_classes=10,
            d_model=128,
            num_heads=4,
            dff=256,
            num_layers=3
        )
        
        x = torch.randn(2, 30, 71)
        output = model(x)
        assert output.shape == (2, 10)

    def test_model_is_nn_module(self, model):
        """模型应是PyTorch nn.Module"""
        assert isinstance(model, torch.nn.Module)

    def test_model_parameters(self, model):
        """模型应有可训练参数"""
        params = list(model.parameters())
        assert len(params) > 0

    def test_positional_encoding(self, model):
        """位置编码形状应正确"""
        seq_len = 30
        d_model = 64
        pe = model.pos_encoding
        assert pe.shape == (1, seq_len, d_model)

    def test_trainable(self, model):
        """模型应可训练"""
        x = torch.randn(4, 30, 71)
        y = torch.tensor([0, 1, 2, 3])
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练一步
        model.train()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # 确认梯度存在
        for param in model.parameters():
            assert param.grad is not None
