# 聆心手语识别系统

## 项目简介
聆心（Lingxin）是一个基于深度学习的实时手语识别系统，支持孤立词和连续手语识别。

聆听心灵的声音，搭建无声与有声世界的桥梁。

## 目录结构

```
Lingxin/
 ├── data/                        # 数据目录
 │   ├── raw/                     # 原始数据（视频/关键点）
 │   │   ├── collected/           # 自采数据
 │   │   │   ├── word_你好/       # 每个词一个文件夹
 │   │   │   │   ├── person1_01.npy
 │   │   │   │   ├── person1_02.npy
 │   │   │   │   └── ...
 │   │   │   └── ...
 │   │   └── ms_asl/              # MS-ASL（可选，用于验证）
 │   └── processed/               # 预处理后的数据
 │       └── csl_isolated/        # 自采孤立词数据
 │
 ├── src/                         # 核心代码
 │   ├── detection/               # 手部检测
 │   │   └── hand_detector.py     # MediaPipe 封装
 │   ├── features/                # 特征工程
 │   │   └── feature_extractor.py # 关键点→特征
 │   ├── models/                  # 模型定义
 │   │   ├── classifiers.py       # SVM/RF/MLP
 │   │   ├── lstm_model.py        # LSTM
 │   │   └── transformer_model.py # Transformer
 │   ├── training/                # 训练逻辑
 │   │   └── trainer.py           # 统一训练接口
 │   └── utils/                   # 工具函数
 │       ├── data_loader.py       # 数据加载
 │       ├── visualization.py     # 可视化
 │       └── metrics.py           # 评估指标
 │
 ├── tools/                       # 工具脚本
 │   ├── collect_data.py          # 数据采集工具 ⭐
 │   ├── preprocess.py            # 数据预处理
 │   ├── train.py                 # 训练入口
 │   ├── evaluate.py              # 评估入口
 │   └── inference.py             # 推理/演示
 │
 ├── api/                         # API 服务
 │   └── app.py                   # Flask/FastAPI
 │
 ├── docs/                        # 文档
 │   ├── architecture.md          # 架构设计
 │   ├── data_collection.md       # 数据采集指南
 │   └── experiments.md           # 实验记录
 │
 ├── tests/                       # 测试
 │   └── ...
 │
 ├── requirements.txt
 ├── .gitignore
 └── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据采集

```bash
python tools/collect_data.py
```

## 数据预处理

```bash
python tools/preprocess.py
```

## 模型训练

```bash
python tools/train.py
```

## 模型评估

```bash
python tools/evaluate.py
```

## 实时推理

```bash
python tools/inference.py
```

## API 服务

```bash
python api/app.py
```

## 文档

详细文档请参考 `docs/` 目录。