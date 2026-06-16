# 06 — 设计模式实践

> **学习目标**：理解项目中设计模式的应用，学会评估何时使用模式、何时保持简单。

## 一、为什么需要设计模式

在项目初期，代码存在以下问题：

1. **代码重复**：`train_lstm()` 和 `train_transformer()` 重复约 80% 的训练逻辑
2. **违反开闭原则**：添加新模型需要修改多处 if-elif 分支
3. **职责不清**：Trainer 类承担了太多职责
4. **配置散落**：路径、超参数、常量散落在各模块中

## 二、项目采用的设计模式

### 2.1 模板方法模式（Template Method）

**位置**：`src/models/base_model.py`

**问题**：LSTM 和 Transformer 的训练流程完全相同（数据加载→训练循环→验证→保存），但代码各写一份。

**解决**：在 `BaseModel` 中定义训练骨架，子类只需实现 `forward()` 方法。

```python
class BaseModel(ABC, nn.Module):
    def train_model(self, train_loader, val_loader, epochs, lr):
        """模板方法：定义训练算法骨架（子类无需重写）"""
        criterion = self._get_criterion()
        optimizer = self._get_optimizer(lr)
        for epoch in range(epochs):
            # 统一训练循环
            ...

    @abstractmethod
    def forward(self, x):
        """子类只需实现模型结构"""
        pass
```

**效果**：

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| LSTM 训练代码 | 60 行 | 0 行（继承） |
| Transformer 训练代码 | 60 行 | 0 行（继承） |
| 新增模型训练代码 | 需写 60 行 | 0 行（继承） |

### 2.2 字典映射（策略模式替代方案）

**位置**：`tools/train.py`

**问题**：大量 if-elif 分支判断模型类型。

**解决**：使用 `MODEL_MAP` 字典，一行查找替代多层分支。

```python
MODEL_MAP = {
    'svm':  {'type': 'classifier',    'model_class': None},
    'rf':   {'type': 'classifier',    'model_class': None},
    'mlp':  {'type': 'classifier',    'model_class': None},
    'lstm': {'type': 'deep_learning', 'model_class': LSTMModel},
    'transformer': {'type': 'deep_learning', 'model_class': TransformerModel},
}

# 使用时
model_info = MODEL_MAP[model_type]  # 一行替代 15 行 if-elif
```

**为什么不用策略模式/工厂模式？**
- 项目只有 5 种模型，类型固定
- 字典映射代码更少、更直观、调试更方便
- 工厂模式会产生额外的类和接口，对当前规模是过度设计

### 2.3 单例配置模式

**位置**：`src/config.py`

**问题**：路径和超参数散落在各文件中，修改需要找多处。

**解决**：模块级 `ProjectConfig` 单例（dataclass），全局唯一配置源。

```python
from src.config import config

# 任何模块中统一访问
data_dir = config.processed_data_dir
epochs = config.default_epochs
model_path = config.get_model_path('lstm')
```

### 2.4 常量集中管理

**位置**：`src/constants.py`

**问题**：`21`、`63`、`171`、`30` 等魔法数字散布各文件。

**解决**：共享常量定义在单一文件中，所有模块引用同一数据源。

```python
from src.constants import RAW_HOLISTIC_FEATURES  # 171 — 而非硬编码数字
```

### 2.5 依赖注入

**位置**：`src/training/trainer.py`

**作用**：训练器不依赖具体模型实现，模型类通过参数传入。

```python
# 训练器不 import LSTMModel 或 TransformerModel
# 模型类由调用方传入
model, accuracy = Trainer.train_deep_learning(X, y, model_class=LSTMModel)
```

**优势**：训练逻辑与模型实现解耦。

## 三、避免过度设计

> **核心原则**：设计模式不是银弹！过度设计会增加复杂度，降低可维护性。

### 过度设计的标志

- 为了"以后扩展"添加大量抽象
- 使用复杂模式解决简单问题（如 5 个模型用了完整工厂+策略模式）
- 创建大量不必要的接口和基类

### 本项目的取舍

| 模式 | 是否使用 | 理由 |
|------|----------|------|
| **模板方法** | ✅ 使用 | LSTM/Transformer 训练流程完全相同 |
| **单例配置** | ✅ 使用 | 全局配置集中管理 |
| 策略模式 | ❌ 未使用 | 字典映射更简单直接 |
| 工厂模式 | ❌ 未使用 | 模型类型固定，不需要动态创建 |

### 重构原则

| 原则 | 含义 |
|------|------|
| **YAGNI** | You Aren't Gonna Need It — 不要为"未来可能需要"而设计 |
| **KISS** | Keep It Simple, Stupid — 保持简单 |
| **DRY** | Don't Repeat Yourself — 消除重复代码 |
| **SOLID** | 单一职责、开闭原则等 |

## 四、项目最终架构

```
tools/train.py
    │
    ├── MODEL_MAP (字典映射) ──→ 消除 if-elif 分支
    │
    └── Trainer (静态方法)
          ├── train_classifier()      ──→ sklearn (SVM/RF/MLP)
          └── train_deep_learning()   ──→ BaseModel (模板方法)
                                              ├── LSTMModel
                                              └── TransformerModel

src/
├── config.py      (单例配置)
├── constants.py   (常量集中)
└── utils/logger.py (日志统一)
```

## 五、扩展指南

添加新模型只需两步，无需修改训练逻辑：

**步骤 1**：创建模型类，继承 `BaseModel`，实现 `forward()`

```python
class GRUModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_shape[1], 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])
```

**步骤 2**：在 `MODEL_MAP` 中注册

```python
MODEL_MAP = {
    ...
    'gru': {'type': 'deep_learning', 'model_class': GRUModel},
}
```

## 六、下一步

- 深入代码实现 → [05-核心代码导读](05-code-guide.md)
- 了解系统架构 → [02-架构设计](02-architecture.md)

---

*最后更新：2026-06-10*
