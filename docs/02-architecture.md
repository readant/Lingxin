# 聆心手语识别系统 - 架构设计

## 一、系统概述

聆心（Lingxin）是一个基于深度学习的实时手语识别系统，旨在搭建无声与有声世界的桥梁。系统采用分层架构设计，具有良好的可扩展性和可维护性。

## 二、项目目录结构

```
Lingxin/                              # 项目根目录
├── api/                              # API服务模块
│   └── app.py                        # Flask API应用入口（含WebSocket）
├── assets/                           # 静态资源
│   └── marked.min.js                 # Markdown解析库
├── data/                             # 数据目录
│   ├── vocab.csv                     # 自建词汇表（50个常用词汇）
│   ├── raw/                          # 原始数据
│   │   └── collected/                # 采集输出（按词汇分目录）
│   └── processed/                    # 预处理后的训练数据
├── docs/                             # 文档目录
│   ├── 01-quickstart.md              # 快速入门
│   ├── 02-architecture.md            # 架构设计
│   ├── 03-data-collection.md         # 数据采集指南
│   ├── 04-training.md                # 模型训练
│   ├── 05-code-guide.md              # 核心代码导读
│   ├── 06-design-patterns.md         # 设计模式实践
│   ├── 07-faq.md                     # 常见问题
│   ├── 08-testing.md                 # 测试指南
│   └── journal/                      # 项目日志
│       ├── development.md            # 开发历程
│       └── experiments.md            # 实验记录
├── learning/                         # 新手学习教程（10阶段）
│   ├── README.md                     # 学习指南
│   └── *.py                          # 各阶段学习脚本
├── models/                           # 模型文件目录
│   ├── hand_landmarker.task          # MediaPipe手部检测模型
│   ├── pose_landmarker_lite.task     # MediaPipe姿态检测模型
│   └── *_model.{pth,pkl}            # 训练好的模型权重
├── src/                              # 核心源代码
│   ├── config.py                     # 统一配置管理（dataclass 单例）
│   ├── constants.py                  # 共享常量定义（骨架连接、维度常量）
│   ├── detection/                    # 关键点检测模块
│   │   └── hand_detector.py          # Hand/Pose/Holistic 检测器
│   ├── features/                     # 特征工程模块
│   │   ├── feature_extractor.py      # 特征提取器（71维）
│   │   └── augmentation.py           # 关键点序列数据增强
│   ├── models/                       # 模型定义模块
│   │   ├── base_model.py             # 模型基类（模板方法模式）
│   │   ├── classifiers.py            # 传统ML模型（SVM/RF/MLP）
│   │   ├── lstm_model.py             # LSTM序列模型
│   │   └── transformer_model.py      # Transformer序列模型
│   ├── training/                     # 训练模块
│   │   └── trainer.py                # 统一训练接口（支持按人员划分）
│   └── utils/                        # 工具函数模块
│       ├── data_loader.py            # 数据加载（人员ID解析、元信息读取）
│       ├── logger.py                 # 统一日志系统
│       ├── metrics.py                # 评估指标计算
│       └── visualization.py          # 可视化工具
├── tests/                            # 测试代码
│   ├── test_augmentation.py          # 数据增强测试
│   ├── test_collect_data.py          # 数据采集测试
│   ├── test_collect_from_video.py    # 视频采集测试
│   ├── test_config.py                # 配置管理测试
│   ├── test_constants.py             # 常量测试
│   └── test_feature_extractor.py     # 特征提取测试
├── tools/                            # 工具脚本
│   ├── collect_data.py               # 摄像头实时采集
│   ├── collect_from_video.py         # 视频文件批量采集
│   ├── preprocess.py                 # 数据预处理（支持按人员划分）
│   ├── train.py                      # 模型训练入口
│   ├── evaluate.py                   # 模型评估入口
│   └── inference.py                  # 实时推理入口
├── web/                              # Web演示界面
│   ├── index.html                    # 首页（学习导航）
│   ├── dashboard.html                # 全流程控制台
│   ├── demo.html                     # 实时手语识别演示
│   ├── docs.html                     # 文档在线查看器
│   ├── resources.html                # 学习资源导航
│   └── static/                       # 静态资源
├── pyproject.toml                    # 项目元数据和构建配置
├── requirements.txt                  # pip 依赖清单
├── environment.yml                   # conda CPU环境文件
├── environment-gpu.yml               # conda GPU环境文件
├── .pre-commit-config.yaml           # Git钩子配置
└── README.md                         # 项目说明
```

## 三、系统架构层次

### 1. 数据层（Data Layer）

负责数据的存储和管理：
- **原始数据**：采集的手语视频和关键点数据
- **词汇表**：
  - `data/vocab.csv`：自建词汇表，50个常用词汇（7大类别）
- **视频文件**：`data/videos/` 目录，存放待处理的视频素材
- **预处理数据**：`data/processed/`，按人员划分的训练/验证/测试集

### 2. 检测层（Detection Layer）

基于 MediaPipe 实现关键点检测：
- **HandDetector**：检测手部21个关键点
- **PoseDetector**：检测上半身15个关键点
- **HolisticDetector**：综合检测器，同时获取手部和姿态数据（171维特征向量）

### 3. 特征层（Feature Layer）

将原始关键点转换为有意义的特征：
- 相对坐标变换
- 手指长度计算
- 关节角度计算
- 时间序列特征提取

### 4. 模型层（Model Layer）

支持多种机器学习和深度学习模型：

| 模型类型 | 模型名称 | 适用场景 |
|----------|----------|----------|
| 传统机器学习 | SVM | 快速推理、资源受限 |
| 传统机器学习 | 随机森林 | 特征重要性分析 |
| 传统机器学习 | MLP | 中等复杂度任务 |
| 深度学习 | LSTM | 时序数据建模 |
| 深度学习 | Transformer | 高精度要求场景 |

### 5. 应用层（Application Layer）

提供面向用户的工具和服务：
- **数据采集**：`tools/collect_data.py`
- **模型训练**：`tools/train.py`
- **模型评估**：`tools/evaluate.py`
- **实时推理**：`tools/inference.py`
- **API服务**：`api/app.py`

## 四、核心数据流

```
摄像头输入 → 关键点检测 → 特征提取 → 模型预测 → 结果输出
     ↓              ↓            ↓          ↓
   视频帧       171维向量    特征向量     手语词汇
```

### 数据采集流程

**方式A：摄像头实时采集**
1. 用户通过 `tools/collect_data.py` 录制手语动作
2. HolisticDetector 实时检测关键点（空格键控制录制）
3. 数据保存为 `{人员ID}_{序号}.npy` + 元信息 JSON

**方式B：视频文件批量采集**
1. 将视频放入 `data/videos/` 目录
2. `tools/collect_from_video.py` 批量处理视频文件
3. 自动提取关键点序列并保存（输出格式与方式A一致）

### 模型训练流程
1. `tools/preprocess.py` 加载原始数据，按人员划分训练/验证/测试集
2. `tools/train.py` 自动检测预划分数据，根据配置选择模型
3. 训练结果保存到 `models/` 目录（`.pkl` 或 `.pth`）

### 实时推理流程
1. 摄像头实时捕获视频帧
2. HandDetector/HolisticDetector 提取关键点
3. FeatureExtractor 计算特征
4. 模型预测并输出识别结果

## 五、关键技术选型

| 分类 | 技术 | 版本 | 选择理由 |
|------|------|------|----------|
| 深度学习框架 | PyTorch | >=2.0.0,<2.12.0 | 灵活易用，社区活跃，适合研究和生产 |
| 关键点检测 | MediaPipe | **>=0.10.33** | 新版Task API，性能优化，支持WebGPU |
| 图像处理 | OpenCV | >=4.8.0 | 功能强大，支持多种图像操作 |
| 机器学习 | scikit-learn | >=1.3.0,<1.5.0 | 成熟稳定，API友好 |
| 数据处理 | NumPy / Pandas | >=1.24.0 / >=2.0.0 | 数值计算和数据处理基础 |
| API框架 | Flask | >=3.0.0 | 轻量级，易于部署 |
| 可视化 | Matplotlib / Seaborn | >=3.7.0 / >=0.12.0 | 功能完善，社区支持好 |
| 配置管理 | Python dataclass | (标准库) | 零依赖，类型安全 |
| 日志系统 | Python logging | (标准库) | 灵活配置，支持文件/控制台输出 |

## 六、设计模式与架构实践

### 1. 模板方法模式（Template Method）
- **位置**：`src/models/base_model.py`
- **作用**：定义深度学习模型的骨架，子类实现具体细节
- **优势**：减少代码重复，统一训练流程

### 2. 策略模式替代方案
- **位置**：`tools/train.py`、`tools/evaluate.py`、`tools/inference.py`
- **实现方式**：使用字典映射（`MODEL_MAP`）替代 if-elif 分支
- **优势**：更简洁，易于扩展新模型

### 3. 依赖注入
- **位置**：`src/training/trainer.py`
- **作用**：训练器不依赖具体模型实现
- **优势**：解耦训练逻辑和模型实现

### 4. 单例配置模式
- **位置**：`src/config.py`
- **实现**：模块级全局 `ProjectConfig` 单例（dataclass）
- **优势**：单一配置来源，避免路径和超参数散落各处

### 5. 常量集中管理
- **位置**：`src/constants.py`
- **作用**：骨架连接、特征维度等共享常量统一维护
- **优势**：消除魔法数字，各模块引用同一数据源

## 七、模块依赖关系

```
tools/ (入口层)
  ├── collect_data.py  ←── HolisticDetector, config, constants
  ├── collect_from_video.py ←── HolisticDetector, config
  ├── preprocess.py    ←── DataLoader, config, logger
  ├── train.py         ←── Trainer, BaseModel, config, logger
  ├── evaluate.py      ←── MetricsCalculator, config
  └── inference.py     ←── HandDetector, FeatureExtractor, config

src/ (核心层)
  ├── config.py         ←── (无依赖，纯配置)
  ├── constants.py      ←── (无依赖，纯常量)
  ├── detection/        ←── mediapipe
  ├── features/         ←── numpy
  ├── models/           ←── torch, sklearn
  ├── training/         ←── models/, utils/, config
  └── utils/            ←── numpy, sklearn, config
```

## 八、扩展建议

### 新增模型支持
1. 在 `src/models/` 目录下创建新模型文件
2. 在 `tools/train.py` 的 `MODEL_MAP` 中添加配置
3. 实现模型的 `train()` 和 `predict()` 方法

### 新增功能模块
1. 在 `src/` 目录下创建新模块目录
2. 编写核心类和方法
3. 在 `src/config.py` 中注册相关配置项
4. 在工具脚本中集成新模块

---

> **作者提示**：本架构设计遵循 KISS（Keep It Simple, Stupid）原则，避免过度设计。在满足需求的前提下，优先选择简单、直观的实现方式。
