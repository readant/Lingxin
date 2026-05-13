"""
BaseModel - 深度学习模型抽象基类（模板方法模式）

本文件定义了深度学习模型的抽象基类，通过模板方法模式统一训练流程。
所有深度学习模型（如LSTM、Transformer）都应继承此类，并实现forward方法。

设计模式：模板方法模式
- 定义算法骨架（训练流程）
- 将具体步骤（模型结构）延迟到子类实现

功能特性：
- 统一的训练循环
- 自动验证
- 模型保存/加载
- 预测接口
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel(ABC, nn.Module):
    """
    深度学习模型抽象基类
    
    所有深度学习模型必须继承此类，并实现抽象方法forward()。
    基类提供了统一的训练流程、验证、预测和保存加载功能。
    """
    
    def __init__(self):
        """初始化基类，调用父类nn.Module的构造函数"""
        super().__init__()
    
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
    
    def train_model(self, train_loader, val_loader=None, epochs=50, lr=0.001):
        """
        模板方法：统一的训练流程
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器. Defaults to None.
            epochs (int, optional): 训练轮数. Defaults to 50.
            lr (float, optional): 学习率. Defaults to 0.001.
        """
        # 获取损失函数（可被子类重写）
        criterion = self._get_criterion()
        # 获取优化器（可被子类重写）
        optimizer = self._get_optimizer(lr)
        
        # 训练主循环
        for epoch in range(epochs):
            # 设置模型为训练模式
            self.train()
            running_loss = 0.0
            
            # 遍历训练批次
            for inputs, labels in train_loader:
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播
                outputs = self(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
                # 累加损失
                running_loss += loss.item() * inputs.size(0)
            
            # 计算平均损失
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
            
            # 如果提供了验证集，则进行验证
            if val_loader:
                val_loss, val_accuracy = self._validate(val_loader, criterion)
                print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
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
        # 设置模型为评估模式
        self.eval()
        val_loss = 0.0
        correct = 0
        
        # 禁用梯度计算（节省内存和计算资源）
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                # 获取预测结果（取最大值的索引）
                _, predicted = torch.max(outputs, 1)
                # 统计正确预测数量
                correct += (predicted == labels).sum().item()
        
        # 计算平均损失和准确率
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
        # 设置模型为评估模式
        self.eval()
        
        # 如果输入是numpy数组，转换为tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # 禁用梯度计算
        with torch.no_grad():
            outputs = self(x)
            # 获取预测结果
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
        self.load_state_dict(torch.load(path))