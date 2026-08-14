# 聆心手语识别系统

![GitHub Stars](https://img.shields.io/github/stars/readant/Lingxin?style=social) ![GitHub Forks](https://img.shields.io/github/forks/readant/Lingxin?style=social) ![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![Status](https://img.shields.io/badge/status-active-success)

## 项目简介

聆心（Lingxin）是一个基于深度学习的实时手语识别系统，支持孤立词手语识别。

- **愿景**：聆听心灵的声音，搭建无声与有声世界的桥梁
- **技术栈**：PyTorch · MediaPipe · scikit-learn · Flask
- **支持模型**：SVM、随机森林、MLP、LSTM、Transformer

---

## 🎯 适用人群与学习目标

### 适用人群

- **手语学习者**：希望通过AI技术辅助手语学习和练习的初学者
- **AI初学者**：对计算机视觉、深度学习感兴趣，想从实战项目入门的开发者
- **公益开发者**：关注无障碍沟通，希望用技术回馈社会的工程师
- **学生与研究者**：需要手语识别方向的课程设计、毕业设计或研究课题

### 学习目标

学完本项目，你将能够：

1. 掌握实时关键点检测技术（MediaPipe Task API）
2. 理解时序数据处理和特征工程的完整流程
3. 独立训练5种不同架构的分类模型
4. 搭建完整的AI应用（从数据采集到Web部署）
5. 学习工程化实践（配置管理、测试、文档）

---

## ✨ 核心功能

| 功能模块 | 说明 |
|----------|------|
| 手部/姿态检测 | MediaPipe 36点关键点检测（Task API） |
| 时序数据采集 | 摄像头录制 + 视频批量提取，支持多人数据 |
| 数据增强 | 平移、缩放、噪声、时间扭曲、遮挡5种策略 |
| 特征工程 | 相对坐标、角度、长度等71维特征提取 |
| 多模型训练 | SVM/随机森林/MLP/LSTM/Transformer 一键切换 |
| 模型评估 | 准确率、F1、混淆矩阵多维评估 |
| 实时推理 | 摄像头实时识别，支持5FPS流畅体验 |
| Web演示 | 浏览器端在线手语识别演示 |
| 新手教程 | 13阶段循序渐进的学习路线 |

---

## 🚀 快速上手

### 1. 环境配置

#### 方法一：使用conda环境文件（推荐）

```bash
# 克隆项目
git clone https://github.com/readant/Lingxin.git
cd Lingxin

# 创建conda环境
conda env create -f environment-gpu.yml

# 激活环境
conda activate lingxin-gpu

# 验证安装
python -c "import torch; import mediapipe; print('✅ 安装成功')"
```

#### 方法二：手动创建conda环境

```bash
# 创建Python 3.10环境
conda create -n lingxin-gpu python=3.10 -y
conda activate lingxin-gpu

# 安装PyTorch（CPU版本）
conda install "pytorch>=2.0.0,<2.12.0" torchvision cpuonly -c pytorch -y

# 安装其他依赖
conda install "scikit-learn>=1.3.0,<1.5.0" pandas "numpy>=1.24.0,<1.27.0" matplotlib seaborn tqdm pillow opencv flask -y

# 安装MediaPipe（新版Task API）
pip install mediapipe>=0.10.33 flask-cors>=4.0.0

# 验证安装
python -c "import torch; import mediapipe; print('✅ 安装成功')"
```

#### 方法三：使用pip安装

```bash
# 克隆项目
git clone https://github.com/readant/Lingxin.git
cd Lingxin

# 安装依赖（依赖以 pyproject.toml 为唯一真相源）
pip install -e ".[dev]"

# 验证安装
python -c "import torch; import mediapipe; print('✅ 安装成功')"
```

### 2. 下载模型

```bash
# 首次使用前必须下载MediaPipe预训练模型
python learning/download_models.py
```

### 3. 完整工作流

#### 步骤一：数据采集

```bash
# 运行数据采集工具
python tools/collect_data.py
```

**采集操作**：
- 输入采集人员ID（如：user001）
- 输入每个词的目标录制数量（建议：50）
- 按 **空格键** 开始/停止录制
- 按 **N/P** 切换上/下一个词汇
- 按 **Q** 查看统计并退出

#### 步骤二：数据预处理

```bash
# 预处理采集的数据
python tools/preprocess.py --input data/raw/collected --output data/processed
```

预处理内容：数据标准化、序列对齐（统一为30帧）、数据集划分（80%/10%/10%）

#### 步骤三：模型训练

```bash
# 选择模型进行训练
python tools/train.py --model lstm --data data/processed
```

| 模型 | 命令参数 | 适用场景 |
|------|----------|----------|
| SVM | `--model svm` | 快速训练、实时推理 |
| 随机森林 | `--model rf` | 特征重要性分析 |
| MLP | `--model mlp` | 中等复杂度任务 |
| LSTM | `--model lstm` | 时序数据建模 |
| Transformer | `--model transformer` | 高精度要求 |

#### 步骤四：模型评估

```bash
# 评估训练好的模型
python tools/evaluate.py --model lstm --checkpoint models/lstm_model.pth
```

评估指标：准确率、精确率、召回率、F1分数、混淆矩阵

#### 步骤五：实时推理

```bash
# 启动实时手语识别
python tools/inference.py --model lstm --checkpoint models/lstm_model.pth
```

打开摄像头，做出手语动作，系统实时输出识别结果。

#### 步骤六：API服务

```bash
# 启动Flask API服务
python api/app.py --model lstm
```

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/predict` | POST | 手语识别（接收base64图片或171维特征） |
| `/api/detect` | POST | 手部关键点检测（返回171维特征） |
| `/api/load_model` | POST | 加载/切换模型 |
| `/api/health` | GET | 健康检查 |
| `/ws/detect` | WebSocket | 实时检测+预测 |

#### 步骤七：Web演示

启动API服务后，访问 http://localhost:5000 体验浏览器端演示：

- **首页** `/` — 项目入口和学习导航
- **实时演示** `/demo` — 在线体验手语识别
- **全流程控制台** `/dashboard` — 数据采集、训练、评估一站式管理

---

## � 项目结构

```
Lingxin/
├── src/                              # 核心源代码
│   ├── config.py                     # 统一配置管理
│   ├── constants.py                  # 共享常量
│   ├── detection/                    # 关键点检测
│   │   └── hand_detector.py          # 手部/姿态/Holistic检测器
│   ├── features/                     # 特征工程
│   │   ├── feature_extractor.py      # 特征提取器（71维）
│   │   └── augmentation.py           # 数据增强模块
│   ├── models/                       # 模型定义
│   │   ├── base_model.py             # 模型基类（模板方法模式）
│   │   ├── lstm_model.py             # LSTM模型
│   │   └── transformer_model.py      # Transformer模型
│   ├── training/                     # 训练模块
│   │   └── trainer.py                # 统一训练接口
│   └── utils/                        # 工具函数
│       ├── data_loader.py            # 数据加载
│       ├── logger.py                 # 统一日志系统
│       ├── metrics.py                # 评估指标
│       └── visualization.py          # 可视化工具
│
├── tools/                            # 命令行工具
│   ├── collect_data.py               # 摄像头数据采集
│   ├── collect_from_video.py         # 视频批量采集
│   ├── preprocess.py                 # 数据预处理
│   ├── train.py                      # 模型训练入口
│   ├── evaluate.py                   # 模型评估入口
│   └── inference.py                  # 实时推理入口
│
├── api/                              # API服务
│   └── app.py                        # Flask API入口（含WebSocket）
│
├── web/                              # Web演示界面
│   ├── index.html                    # 首页（学习导航）
│   ├── demo.html                     # 实时手语识别演示
│   ├── dashboard.html                # 全流程控制台
│   └── static/                       # 静态资源
│
├── learning/                         # 新手学习教程（13个阶段）
│   ├── README.md                     # 学习指南和路线图
│   ├── download_models.py           # 模型下载脚本
│   └── 00-13阶段教程.py              # 循序渐进学习
│
├── tests/                            # 单元测试
├── data/                             # 数据目录
├── models/                           # 模型文件目录
├── docs/                             # 详细技术文档
├── pyproject.toml                    # 项目配置（依赖唯一真相源）
└── environment-gpu.yml               # conda GPU环境重建手册
```

---

## 📚 详细文档

| 文档 | 适合人群 | 内容简介 |
|------|----------|----------|
| [📖 文档中心](docs/README.md) | 所有人 | 文档导航、推荐学习路径 |
| [01-快速入门](docs/guide/01-quickstart.md) | 初学者 | 项目简介、环境搭建、5分钟体验 |
| [02-架构设计](docs/design/02-architecture.md) | 开发者 | 系统架构、模块设计、技术选型 |
| [03-数据采集指南](docs/usage/03-data-collection.md) | 采集者 | 摄像头与视频批量采集全流程 |
| [04-模型训练](docs/usage/04-training.md) | 工程师 | 训练、评估、推理完整流程 |
| [05-核心代码导读](docs/design/05-code-guide.md) | 贡献者 | 源码结构、模块详解、扩展指南 |
| [06-设计模式实践](docs/design/06-design-patterns.md) | 进阶 | 模板方法、字典映射实践 |
| [07-常见问题](docs/dev/07-faq.md) | 所有人 | 环境/数据/模型问题排查 |
| [08-测试指南](docs/dev/08-testing.md) | 贡献者 | pytest使用、测试编写 |
| [🎓 学习教程](learning/README.md) | 零基础 | 13阶段循序渐进学习路线 |

---

## 📝 更新日志

### v0.3.0 (2026-06-21)
- 前端重构：统一导航组件，新增全流程控制台
- 实时推理优化：优化帧率至5FPS
- 文档更新：新增配置管理学习指南

### v0.2.0 (2026-06-16)
- 项目工程化：pyproject.toml、pre-commit hooks
- learning教程：新增配置管理/训练流程/数据增强3个教程
- Web演示：新增在线手语识别和MediaPipe WASM资源

更多历史版本请查看 [开发历程](docs/journal/development.md)

---

## 📄 许可证

本项目基于 MIT License 开源，欢迎提交 Issue 和 Pull Request！

[![Star History](https://api.star-history.com/svg?repos=readant/Lingxin&type=svg)](https://star-history.com/#readant/Lingxin&Date)
