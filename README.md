# 聆心手语识别系统

## 项目简介

聆心（Lingxin）是一个基于深度学习的实时手语识别系统，支持孤立词手语识别。

- **愿景**：聆听心灵的声音，搭建无声与有声世界的桥梁
- **技术栈**：PyTorch + MediaPipe + scikit-learn
- **支持模型**：SVM、随机森林、MLP、LSTM、Transformer

## 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 手部关键点检测 | ✅ | MediaPipe 21点手部检测 |
| 姿态关键点检测 | ✅ | MediaPipe 上半身15点检测 |
| 时序数据采集 | ✅ | 支持录制关键点序列 |
| 特征提取 | ✅ | 相对坐标、角度、长度等特征 |
| 模型训练 | ✅ | 支持5种模型 |
| 模型评估 | ✅ | 多指标评估 + 混淆矩阵 |
| 实时推理 | ✅ | 摄像头实时识别 |

## 目录结构

```
Lingxin/
├── api/                              # API服务
│   └── app.py                        # Flask API入口
│
├── data/                             # 数据目录
│   ├── vocab.csv                     # 词汇表（50个常用词汇）
│   ├── raw/                          # 原始数据
│   │   └── collected/                # 自采数据（按词汇分类）
│   └── processed/                    # 预处理后的数据
│
├── docs/                             # 文档目录
│   ├── learning/                     # 新手学习笔记
│   ├── architecture.md               # 架构设计文档
│   ├── data_collection.md            # 数据采集指南
│   ├── development.md               # 开发记录
│   └── experiments.md               # 实验记录
│
├── src/                              # 核心源代码
│   ├── detection/                   # 关键点检测
│   │   └── hand_detector.py         # 手部/姿态/Holistic检测器
│   ├── features/                    # 特征工程
│   │   └── feature_extractor.py     # 特征提取器
│   ├── models/                      # 模型定义
│   │   ├── base_model.py           # 模型基类（模板方法模式）
│   │   ├── classifiers.py          # SVM/随机森林/MLP
│   │   ├── lstm_model.py           # LSTM模型
│   │   └── transformer_model.py   # Transformer模型
│   ├── training/                   # 训练模块
│   │   └── trainer.py              # 统一训练接口
│   └── utils/                      # 工具函数
│       ├── data_loader.py          # 数据加载
│       ├── metrics.py              # 评估指标
│       └── visualization.py         # 可视化工具
│
├── tools/                            # 工具脚本
│   ├── collect_data.py              # 数据采集工具
│   ├── preprocess.py               # 数据预处理工具
│   ├── train.py                    # 模型训练入口
│   ├── evaluate.py                 # 模型评估入口
│   └── inference.py                # 实时推理入口
│
├── tests/                            # 测试目录（待完善）
├── requirements.txt                  # pip依赖清单
├── environment.yml                  # conda环境文件
├── .gitignore
└── README.md
```

## 环境配置（推荐使用conda）

### 方法一：使用conda环境文件（推荐）

```bash
# 1. 克隆项目
git clone git@github.com:readant/Lingxin.git
cd Lingxin

# 2. 创建conda环境
conda env create -f environment.yml

# 3. 激活环境
conda activate lingxin

# 4. 验证安装
python -c "import torch; import mediapipe; print('安装成功')"
```

### 方法二：手动创建conda环境

```bash
# 1. 克隆项目
git clone git@github.com:readant/Lingxin.git
cd Lingxin

# 2. 创建conda环境（Python 3.9-3.11均可）
conda create -n lingxin python=3.10 -y

# 3. 激活环境
conda activate lingxin

# 4. 安装PyTorch（CPU版本）
conda install pytorch torchvision cpuonly -c pytorch -y

# 5. 安装其他依赖
pip install mediapipe==0.10.5 opencv-python scikit-learn pandas numpy matplotlib seaborn tqdm pillow flask flask-cors

# 6. 验证安装
python -c "import torch; import mediapipe; print('安装成功')"
```

### 方法三：使用pip安装（仅限已有Python环境）

```bash
# 1. 克隆项目
git clone git@github.com:readant/Lingxin.git
cd Lingxin

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装兼容版本的MediaPipe（重要！）
pip install mediapipe==0.10.5
```

### 依赖版本说明

| 依赖包 | 推荐版本 | 说明 |
|--------|----------|------|
| Python | 3.9-3.11 | 不支持Python 3.12+ |
| mediapipe | **0.10.5** | 新版本API不兼容，请务必使用此版本 |
| torch | >=2.0.0 | 深度学习框架 |
| opencv-python | >=4.8.0 | 图像处理 |
| scikit-learn | >=1.3.0 | 机器学习模型 |

## 快速开始

### 1. 数据采集

```bash
# 激活环境
conda activate lingxin

# 运行数据采集工具
python tools/collect_data.py
```

**采集步骤**：
1. 输入采集人员ID（如：user001）
2. 输入每个词的目标录制数量（建议：30）
3. 按 **空格键** 开始/停止录制
4. 按 **N/P** 切换上/下一个词汇
5. 按 **Q** 查看统计并退出

### 2. 数据预处理

```bash
# 预处理采集的数据
python tools/preprocess.py --input data/raw/collected --output data/processed
```

预处理内容：
- 数据标准化（归一化）
- 序列对齐（统一为30帧）
- 数据集划分（训练80%/验证10%/测试10%）

### 3. 模型训练

```bash
# 训练模型（支持：svm, rf, mlp, lstm, transformer）
python tools/train.py --model lstm --data data/processed
```

**支持的模型**：

| 模型 | 命令 | 适用场景 |
|------|------|----------|
| SVM | `--model svm` | 快速训练、实时推理 |
| 随机森林 | `--model rf` | 特征重要性分析 |
| MLP | `--model mlp` | 中等复杂度任务 |
| LSTM | `--model lstm` | 时序数据建模 |
| Transformer | `--model transformer` | 高精度要求 |

### 4. 模型评估

```bash
# 评估模型
python tools/evaluate.py --model lstm --checkpoint models/lstm_model.pth
```

评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵

### 5. 实时推理

```bash
# 启动实时推理
python tools/inference.py --model lstm --checkpoint models/lstm_model.pth
```

**操作说明**：
- 摄像头实时显示检测画面
- 做出手语动作
- 系统实时输出识别结果

### 6. API服务

```bash
# 启动Flask API服务
python api/app.py
```

API接口：
- `POST /predict` - 手语识别
- `GET /models` - 获取可用模型列表
- `POST /train` - 触发模型训练

## 项目流程图

```
┌─────────────────────────────────────────────────────────────┐
│                        数据流程                              │
└─────────────────────────────────────────────────────────────┘

  数据采集               预处理              训练              推理
    │                    │                 │                 │
    ▼                    ▼                 ▼                 ▼
┌──────────┐        ┌──────────┐     ┌──────────┐      ┌──────────┐
│摄像头输入│   →    │数据标准化│  →  │模型训练   │  →   │实时预测   │
│          │        │序列对齐   │     │          │      │          │
│空格录制  │        │数据划分   │     │训练+验证  │      │摄像头输入│
└──────────┘        └──────────┘     └──────────┘      └──────────┘
    │                    │                 │                 │
    ▼                    ▼                 ▼                 ▼
┌──────────┐        ┌──────────┐     ┌──────────┐      ┌──────────┐
│.npy文件  │        │训练数据  │     │模型文件  │      │识别结果  │
│(171维×帧)│        │(71维×30) │     │(.pth)    │      │(词汇)    │
└──────────┘        └──────────┘     └──────────┘      └──────────┘
```

## 数据格式说明

### 原始数据（171维）

```
每帧数据：171维向量
├── 左手关键点：63维（21点 × 3坐标）
├── 右手关键点：63维（21点 × 3坐标）
└── 姿态关键点：45维（15点 × 3坐标）
```

### 预处理后数据（71维）

```
每帧数据：71维特征向量
├── 相对坐标：30维（相对于手腕的偏移）
├── 手指长度：10维（5根手指的长度）
└── 关节角度：31维（关节间的角度）
```

### 文件命名规范

```
# 数据文件
{人员ID}_{序号}.npy
示例：user001_001.npy

# 模型文件
{模型名}_model.pth
示例：lstm_model.pth
```

## 常见问题

### Q1: MediaPipe导入报错

**错误**：`AttributeError: module 'mediapipe' has no attribute 'solutions'`

**解决方案**：
```bash
pip install mediapipe==0.10.5
```

### Q2: 模块导入失败

**错误**：`ModuleNotFoundError: No module named 'src'`

**解决方案**：工具脚本已自动添加项目路径，无需手动配置。

### Q3: 中文显示乱码

**现象**：界面显示"???"而非中文

**解决方案**：系统已自动加载Windows中文字体（微软雅黑/黑体），无需额外配置。

### Q4: 摄像头无法打开

**检查项**：
1. 摄像头是否被其他程序占用
2. 摄像头驱动是否正常
3. OpenCV是否正确安装：`pip install opencv-python`

## 词汇表

项目内置50个常用手语词汇：

| 类别 | 词汇数量 | 示例 |
|------|----------|------|
| 问候 | 8 | 你好、谢谢、再见 |
| 人称 | 6 | 我、你、他 |
| 数字 | 10 | 一、二、三 |
| 家庭 | 6 | 爸爸、妈妈 |
| 动作 | 8 | 吃、喝、走 |
| 形容词 | 6 | 好、坏、大 |
| 场景 | 6 | 学校、家 |

详见：`data/vocab.csv`

## 参考资源

- [MediaPipe手部检测](https://google.github.io/mediapipe/solutions/hands)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [scikit-learn机器学习](https://scikit-learn.org/stable/)
- [Runoob设计模式](https://www.runoob.com/python-design-pattern/python-design-pattern-intro.html)

## 文档导航

| 文档 | 内容 |
|------|------|
| [架构设计](docs/architecture.md) | 系统架构、模块设计、技术选型 |
| [数据采集指南](docs/data_collection.md) | 采集工具使用、常见问题 |
| [开发记录](docs/development.md) | 项目历程、问题解决 |
| [实验记录](docs/experiments.md) | 模型对比、参数调优 |
| [学习笔记](docs/learning/) | 新手学习资料 |

## 更新日志

- **2026-05-15**: 完善README文档，重写环境配置指南
- **2026-05-14**: 完成代码注释完善
- **2026-05-13**: 应用设计模式，优化代码结构
- **2026-05-10**: 完成数据采集工具重写

## 贡献指南

欢迎提交Issue和Pull Request！

## 开源协议

MIT License