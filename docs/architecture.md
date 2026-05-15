# 聆心手语识别系统 - 架构设计

## 一、系统概述

聆心（Lingxin）是一个基于深度学习的实时手语识别系统，旨在搭建无声与有声世界的桥梁。系统采用分层架构设计，具有良好的可扩展性和可维护性。

## 二、项目目录结构

```
Lingxin/                              # 项目根目录
├── api/                              # API服务模块
│   └── app.py                        # Flask API应用入口
├── data/                             # 数据目录
│   └── vocab.csv                     # 手语词汇表（50个常用词汇）
├── docs/                             # 文档目录
│   └── learning/                     # 新手学习笔记
├── src/                              # 核心源代码
│   ├── detection/                    # 关键点检测模块
│   │   └── hand_detector.py          # 手部和姿态检测器
│   ├── features/                     # 特征工程模块
│   │   └── feature_extractor.py      # 特征提取器
│   ├── models/                       # 模型定义模块
│   │   ├── base_model.py             # 模型基类（模板方法模式）
│   │   ├── classifiers.py            # 传统ML模型（SVM/RF/MLP）
│   │   ├── lstm_model.py             # LSTM序列模型
│   │   └── transformer_model.py      # Transformer序列模型
│   ├── training/                     # 训练模块
│   │   └── trainer.py                # 统一训练接口
│   └── utils/                        # 工具函数模块
│       ├── data_loader.py            # 数据加载工具
│       ├── metrics.py                # 评估指标计算
│       └── visualization.py          # 可视化工具
├── tools/                            # 工具脚本
│   ├── collect_data.py               # 数据采集工具
│   ├── preprocess.py                 # 数据预处理工具
│   ├── train.py                      # 模型训练入口
│   ├── evaluate.py                   # 模型评估入口
│   └── inference.py                  # 实时推理入口
├── requirements.txt                  # 依赖清单
└── README.md                         # 项目说明
```

## 三、系统架构层次

### 1. 数据层（Data Layer）

负责数据的存储和管理：
- **原始数据**：采集的手语视频和关键点数据
- **词汇表**：`data/vocab.csv`，包含50个常用手语词汇
- **预处理数据**：经过特征提取和标准化的训练数据

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
1. 用户通过 `tools/collect_data.py` 录制手语动作
2. HolisticDetector 实时检测关键点
3. 数据保存为 `.npy` 格式的时序序列

### 模型训练流程
1. `tools/preprocess.py` 加载原始数据并预处理
2. `tools/train.py` 根据配置选择模型进行训练
3. 训练结果保存到模型目录

### 实时推理流程
1. 摄像头实时捕获视频帧
2. HandDetector/HolisticDetector 提取关键点
3. FeatureExtractor 计算特征
4. 模型预测并输出识别结果

## 五、关键技术选型

| 分类 | 技术 | 版本 | 选择理由 |
|------|------|------|----------|
| 深度学习框架 | PyTorch | >=2.0.0 | 灵活易用，社区活跃，适合研究和生产 |
| 关键点检测 | MediaPipe | 0.10.5 | 轻量级、实时性好、准确率高 |
| 图像处理 | OpenCV | >=4.8.0 | 功能强大，支持多种图像操作 |
| 机器学习 | scikit-learn | >=1.3.0 | 成熟稳定，API友好 |
| API框架 | Flask | >=3.0.0 | 轻量级，易于部署 |
| 可视化 | Matplotlib | >=3.7.0 | 功能完善，社区支持好 |

## 六、设计模式应用

### 1. 模板方法模式（Template Method）
- **位置**：`src/models/base_model.py`
- **作用**：定义深度学习模型的骨架，子类实现具体细节
- **优势**：减少代码重复，统一训练流程

### 2. 策略模式替代方案
- **位置**：`tools/train.py`、`tools/evaluate.py`、`tools/inference.py`
- **实现方式**：使用字典映射（`MODEL_CONFIG`）替代 if-elif 分支
- **优势**：更简洁，易于扩展新模型

### 3. 依赖注入
- **位置**：`src/training/trainer.py`
- **作用**：训练器不依赖具体模型实现
- **优势**：解耦训练逻辑和模型实现

## 七、扩展建议

### 新增模型支持
1. 在 `src/models/` 目录下创建新模型文件
2. 在 `tools/train.py` 的 `MODEL_CONFIG` 中添加配置
3. 实现模型的 `train()` 和 `predict()` 方法

### 新增功能模块
1. 在 `src/` 目录下创建新模块目录
2. 编写核心类和方法
3. 在工具脚本中集成新模块

---

> **作者提示**：本架构设计遵循 KISS（Keep It Simple, Stupid）原则，避免过度设计。在满足需求的前提下，优先选择简单、直观的实现方式。