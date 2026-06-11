# 05 — 核心代码导读

> **学习目标**：理解源码结构和核心模块实现，能够基于现有代码进行扩展。

## 一、源码结构总览

```
src/
├── config.py              # 统一配置管理（dataclass 单例）
├── constants.py           # 共享常量（骨架连接、维度常量）
├── detection/             # 关键点检测
│   └── hand_detector.py   # HandDetector / PoseDetector / HolisticDetector
├── features/              # 特征工程
│   ├── feature_extractor.py  # 71维特征提取
│   └── augmentation.py       # 关键点序列数据增强
├── models/                # 模型定义
│   ├── base_model.py         # 抽象基类（模板方法模式）
│   ├── classifiers.py        # SVM / 随机森林 / MLP
│   ├── lstm_model.py         # LSTM 序列模型
│   └── transformer_model.py  # Transformer 序列模型
├── training/              # 训练逻辑
│   └── trainer.py            # 统一训练接口
└── utils/                 # 工具函数
    ├── data_loader.py        # 数据加载（人员ID解析）
    ├── logger.py             # 统一日志系统
    ├── metrics.py            # 评估指标
    └── visualization.py      # 可视化工具
```

## 二、核心模块详解

### 2.1 配置管理 — `src/config.py`

集中管理所有路径、超参数和配置项，避免硬编码。

```python
from src.config import config

# 路径访问
print(config.raw_data_dir)        # data/raw/collected
print(config.processed_data_dir)  # data/processed/csl_isolated

# 训练参数
print(config.default_epochs)      # 50
print(config.default_batch_size)  # 32

# 模型路径
path = config.get_model_path('lstm')  # models/lstm_model.pth
```

**设计要点**：使用 Python `dataclass`，模块级全局单例，`__post_init__` 自动推导派生路径。

### 2.2 常量定义 — `src/constants.py`

项目所有维度常量、骨架连接的唯一数据源：

```python
from src.constants import (
    RAW_HOLISTIC_FEATURES,   # 171
    EXTRACTED_FEATURE_DIMS,  # 71
    DEFAULT_SEQUENCE_LENGTH, # 30
    HAND_CONNECTIONS,        # 手部骨架连接
)
```

### 2.3 关键点检测 — `src/detection/hand_detector.py`

基于 MediaPipe Task API 实现三种检测器：

| 检测器 | 关键点数 | 输出维度 | 用途 |
|--------|----------|----------|------|
| HandDetector | 21点/手 | 63维/手 | 手部关键点 |
| PoseDetector | 15点 | 45维 | 上半身姿态 |
| HolisticDetector | 21×2 + 15 | **171维** | 综合检测（主力） |

```python
from src.detection.hand_detector import HolisticDetector

detector = HolisticDetector(min_detection_confidence=0.5)
results = detector.detect(frame)  # 返回 (left_hand, right_hand, pose) 关键点
```

### 2.4 特征提取 — `src/features/feature_extractor.py`

将 171 维原始关键点转换为 71 维特征向量：

| 特征类型 | 维度 | 计算方式 |
|----------|------|----------|
| 相对坐标 | 63 (21×3) | 所有点减去手腕坐标 |
| 手指长度 | 4 | 4根手指的欧氏距离（指尖→指根） |
| 关节角度 | 4 | 向量点积 → arccos 计算角度 |

```
原始关键点(171维) → 相对化处理 → 手指长度 → 关节角度 → 特征向量(71维)
```

### 2.5 数据增强 — `src/features/augmentation.py`

对关键点序列应用随机变换，增加数据多样性：

| 增强操作 | 说明 | 参数 |
|----------|------|------|
| 随机平移 | 微调关键点位置 | translate_range=0.02 |
| 随机缩放 | 模拟不同距离 | scale_range=(0.9, 1.1) |
| 高斯噪声 | 模拟检测误差 | noise_std=0.01 |
| 时间扭曲 | 模拟速度变化 | time_warp_sigma=3.0 |
| 随机遮挡 | 模拟遮挡 | dropout_prob=0.05 |

```python
from src.features.augmentation import KeypointAugmenter

augmenter = KeypointAugmenter(p=0.5)
augmented = augmenter(sequence)  # 每个操作独立以 p=0.5 概率应用
```

### 2.6 模型基类 — `src/models/base_model.py`

通过**模板方法模式**统一定义深度学习模型的训练骨架：

```python
class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        """子类必须实现：定义模型结构"""
        pass

    def train_model(self, train_loader, val_loader, epochs, lr):
        """模板方法：统一训练流程（子类无需重写）"""
        # 准备 → 训练循环 → 验证 → 早停
```

**好处**：消除 LSTM 和 Transformer 的重复代码约 150 行。

### 2.7 模型实现

#### LSTM (`lstm_model.py`)

```
输入 (batch, 30, 71) → LSTM(128, 2层, dropout=0.2) → 取最后一帧 → FC(128→64) → FC(64→num_classes)
```

#### Transformer (`transformer_model.py`)

```
输入 → 线性嵌入 → 位置编码 → Transformer编码器(4层, 4头) → 全局池化 → FC → 输出
```

#### 传统 ML 模型 (`classifiers.py`)

通过 `Trainer.train_classifier()` 统一调用 scikit-learn 的 SVM/RF/MLP。

### 2.8 训练器 — `src/training/trainer.py`

```python
from src.training.trainer import Trainer

# 传统 ML 模型
result, accuracy = Trainer.train_classifier(X, y, model_type='svm')

# 深度学习模型
model, accuracy = Trainer.train_deep_learning(X, y, model_class=LSTMModel)
```

## 三、训练入口 — `tools/train.py`

使用字典映射消除 if-elif 分支：

```python
MODEL_MAP = {
    'svm':  {'type': 'classifier',    'model_class': None},
    'rf':   {'type': 'classifier',    'model_class': None},
    'mlp':  {'type': 'classifier',    'model_class': None},
    'lstm': {'type': 'deep_learning', 'model_class': LSTMModel},
    'transformer': {'type': 'deep_learning', 'model_class': TransformerModel},
}
```

自动检测数据是否已按 train/val/test 预划分。

## 四、日志系统 — `src/utils/logger.py`

替代散落的 `print()` 调用：

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("开始训练")
logger.warning("数据量不足")
logger.error("模型加载失败")
```

## 五、数据加载器 — `src/utils/data_loader.py`

支持自动解析文件名中的人员 ID：

```python
from src.utils.data_loader import DataLoader, parse_person_id

# parse_person_id('J_001.npy') → 'J'
# parse_person_id('CSL_001.npy') → 'CSL'
# parse_person_id('user001_005.npy') → 'user001'

loader = DataLoader(config.raw_data_dir)
data = loader.load_all()  # 返回按词汇和人员组织的数据字典
```

## 六、扩展指南

### 添加新模型（以 GRU 为例）

**步骤 1**：创建 `src/models/gru_model.py`

```python
import torch.nn as nn
from src.models.base_model import BaseModel

class GRUModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_shape[1], 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])
```

**步骤 2**：在 `tools/train.py` 中注册

```python
from src.models.gru_model import GRUModel

MODEL_MAP = {
    ...
    'gru': {'type': 'deep_learning', 'model_class': GRUModel},
}
```

**无需修改其他代码！**

### 添加新特征提取方法

在 `src/features/` 下创建新提取器，然后在 `config.py` 中可添加相关配置项。

## 七、代码调用关系

```
tools/train.py (入口)
    │
    ├── MODEL_MAP ──→ 模型配置
    │
    └── Trainer
          ├── train_classifier()  ──→ sklearn (SVM/RF/MLP)
          └── train_deep_learning() ──→ BaseModel.train_model() (模板方法)
                                            ├── LSTMModel.forward()
                                            └── TransformerModel.forward()
```

## 八、下一步

- 学习设计模式实践 → [06-设计模式实践](06-design-patterns.md)
- 了解系统架构 → [02-架构设计](02-architecture.md)
- 遇到问题？→ [07-常见问题](07-faq.md)

---

*最后更新：2026-06-10*
