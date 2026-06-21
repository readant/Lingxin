# 聆心手语识别系统

## 项目简介

聆心（Lingxin）是一个基于深度学习的实时手语识别系统，支持孤立词手语识别。

- **愿景**：聆听心灵的声音，搭建无声与有声世界的桥梁
- **技术栈**：PyTorch + MediaPipe + scikit-learn
- **支持模型**：SVM、随机森林、MLP、LSTM、Transformer

## 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 手部关键点检测 | ✅ | MediaPipe 21点手部检测（Task API） |
| 姿态关键点检测 | ✅ | MediaPipe 上半身15点检测（Task API） |
| 时序数据采集 | ✅ | 摄像头录制 + 视频批量提取 |
| 数据增强 | ✅ | 平移、缩放、噪声、时间扭曲、遮挡 |
| 特征提取 | ✅ | 相对坐标、角度、长度等特征 |
| 模型训练 | ✅ | 支持5种模型，按人员划分数据集 |
| 模型评估 | ✅ | 多指标评估 + 混淆矩阵 |
| 实时推理 | ✅ | 摄像头实时识别 |
| Web演示 | ✅ | 浏览器在线手语识别演示 |
| 新手教程 | ✅ | 10阶段循序渐进学习路线 |

## 目录结构

```
Lingxin/
├── api/                              # API服务
│   └── app.py                        # Flask API入口（含WebSocket实时推理）
│
├── assets/                           # 静态资源
│   └── marked.min.js                 # Markdown解析库（文档查看器用）
│
├── data/                             # 数据目录
│   ├── vocab.csv                     # 词汇表（50个常用词汇）
│   ├── raw/                          # 原始数据
│   │   └── collected/                # 自采数据（按词汇分类）
│   └── processed/                    # 预处理后的数据
│
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
│
├── learning/                         # 新手学习教程（10个阶段）
│   ├── README.md                     # 学习指南和路线图
│   ├── download_models.py            # 模型下载脚本
│   ├── 00_env_setup.py              # 环境检查
│   ├── 01_python_basics.py          # Python基础
│   ├── 02_opencv_*.py               # OpenCV图像/视频处理
│   ├── 03_mediapipe_intro.py        # MediaPipe入门
│   ├── 04_hand_detection_*.py       # 手部检测
│   ├── 05_pose_detection.py         # 姿态检测
│   ├── 06_numpy_*.py                # NumPy数据处理
│   ├── 07_feature_extraction.py     # 特征工程
│   ├── 08_svm_intro.py              # SVM入门
│   ├── 09_lstm_intro.py             # LSTM入门
│   └── 10_data_collection.py        # 数据采集实战
│
├── models/                           # 模型文件
│   ├── hand_landmarker.task          # MediaPipe手部检测模型
│   ├── pose_landmarker_lite.task     # MediaPipe姿态检测模型
│   └── *_model.{pth,pkl}            # 训练好的模型权重
│
├── src/                              # 核心源代码
│   ├── config.py                     # 统一配置管理
│   ├── constants.py                  # 共享常量
│   ├── detection/                    # 关键点检测
│   │   └── hand_detector.py         # 手部/姿态/Holistic检测器
│   ├── features/                     # 特征工程
│   │   ├── feature_extractor.py     # 特征提取器
│   │   └── augmentation.py          # 数据增强模块
│   ├── models/                       # 模型定义
│   │   ├── base_model.py           # 模型基类（模板方法模式）
│   │   ├── classifiers.py          # SVM/随机森林/MLP
│   │   ├── lstm_model.py           # LSTM模型
│   │   └── transformer_model.py   # Transformer模型
│   ├── training/                     # 训练模块
│   │   └── trainer.py              # 统一训练接口
│   └── utils/                        # 工具函数
│       ├── data_loader.py           # 数据加载
│       ├── logger.py                # 统一日志系统
│       ├── metrics.py               # 评估指标
│       └── visualization.py         # 可视化工具
│
├── tests/                            # 测试代码
│   ├── test_augmentation.py          # 数据增强测试
│   ├── test_collect_data.py          # 数据采集测试
│   ├── test_collect_from_video.py    # 视频采集测试
│   ├── test_config.py                # 配置管理测试
│   ├── test_constants.py             # 常量测试
│   └── test_feature_extractor.py     # 特征提取测试
│
├── tools/                            # 工具脚本
│   ├── collect_data.py              # 摄像头数据采集工具
│   ├── collect_from_video.py        # 视频批量采集工具
│   ├── preprocess.py               # 数据预处理工具
│   ├── train.py                    # 模型训练入口
│   ├── evaluate.py                 # 模型评估入口
│   └── inference.py                # 实时推理入口
│
├── web/                              # Web演示界面
│   ├── index.html                    # 首页（学习导航）
│   ├── dashboard.html                # 全流程控制台
│   ├── demo.html                     # 实时手语识别演示
│   ├── docs.html                     # 文档在线查看器
│   ├── resources.html                # 学习资源导航
│   ├── static/                       # 静态资源
│   │   ├── css/                      # 样式文件
│   │   └── js/                       # JavaScript文件
│   │       ├── app.js                # 核心应用逻辑
│   │       ├── nav.js                # 统一导航组件
│   │       ├── chart.js              # 图表组件
│   │       └── onboarding.js         # 新手引导
│   └── mediapipe/                    # MediaPipe WASM资源
│
├── pyproject.toml                    # 项目元数据和构建配置
├── requirements.txt                  # pip依赖清单
├── environment.yml                   # conda CPU环境文件
├── environment-gpu.yml               # conda GPU环境文件
├── .pre-commit-config.yaml           # Git钩子配置
├── .gitignore
└── README.md
```

## 环境配置（推荐使用conda）

项目提供两种环境配置：
- **`environment.yml`** — CPU版本（默认，适合开发和学习）
- **`environment-gpu.yml`** — GPU版本（NVIDIA CUDA，适合模型训练加速）

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

### 方法二：手动创建conda环境（推荐）

```bash
# 1. 克隆项目
git clone git@github.com:readant/Lingxin.git
cd Lingxin

# 2. 创建conda环境（Python 3.9-3.11均可）
conda create -n lingxin python=3.10 -y

# 3. 激活环境
conda activate lingxin

# 4. 安装PyTorch（CPU版本，指定版本范围）
conda install "pytorch>=2.0.0,<2.12.0" torchvision cpuonly -c pytorch -y

# 5. 安装其他依赖（优先使用conda，指定版本范围）
conda install "scikit-learn>=1.3.0,<1.5.0" pandas "numpy>=1.24.0,<1.27.0" matplotlib seaborn tqdm pillow opencv flask -y

# 6. 安装必须用pip的包（mediapipe使用新版Task API）
pip install mediapipe>=0.10.33 flask-cors>=4.0.0

# 7. 验证安装
python -c "import torch; import mediapipe; print('安装成功')"
```

### 方法三：使用pip安装（仅限已有Python环境）

```bash
# 1. 克隆项目
git clone git@github.com:readant/Lingxin.git
cd Lingxin

# 2. 安装依赖（requirements.txt已包含所有版本限制）
pip install -r requirements.txt

# 3. 验证安装
python -c "import torch; import mediapipe; print('安装成功')"
```

### 依赖版本说明

| 依赖包 | 推荐版本 | 说明 |
|--------|----------|------|
| Python | 3.9-3.11 | 不支持Python 3.12+ |
| mediapipe | **>=0.10.33** | 新版Task API，性能优化，支持WebGPU |
| torch | >=2.0.0,<2.12.0 | PyTorch深度学习框架 |
| torchvision | >=0.15.0 | PyTorch视觉工具 |
| opencv-python | >=4.8.0 | 图像处理 |
| scikit-learn | >=1.3.0,<1.5.0 | 传统机器学习模型（SVM、随机森林、MLP） |
| numpy | >=1.24.0,<1.27.0 | 数值计算 |
| pandas | >=2.0.0 | 数据处理 |
| matplotlib | >=3.7.0 | 数据可视化 |
| seaborn | >=0.12.0 | 统计数据可视化 |
| tqdm | >=4.65.0 | 进度条 |
| pillow | >=10.0.0 | 图像处理（中文显示） |
| flask | >=3.0.0 | Web框架 |
| flask-cors | >=4.0.0 | 跨域支持 |

### MediaPipe模型下载（新版API必需）

**首次使用前需要下载模型文件**：

```bash
# 下载MediaPipe预训练模型
python learning/download_models.py
```

模型文件会保存到 `models/` 目录：
- `hand_landmarker.task` - 手部检测模型（约10MB）
- `pose_landmarker_lite.task` - 姿态检测模型（约20MB）

### API变更说明（MediaPipe升级）

**旧版API（0.10.5及以下）**：使用`mp.solutions`模块
```python
hands = mp.solutions.hands.Hands(...)
results = hands.process(image)
```

**新版API（0.10.33+）**：使用`mp.tasks`模块（Task API）
```python
options = mp.tasks.vision.HandLandmarkerOptions(...)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
results = detector.detect(mp_image)
```

### 升级优势

| 特性 | 旧版 (0.10.5) | 新版 (0.10.33+) |
|------|---------------|-----------------|
| **推理速度** | 一般 | ✅ 优化 |
| **Web部署** | 一般 | ✅ WebGPU支持 |
| **API设计** | 旧版solutions | ✅ 现代Task API |
| **LLM支持** | 不支持 | ✅ 支持 |
| **内存占用** | 一般 | ✅ 优化 |

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
2. 输入每个词的目标录制数量（建议：50）
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
python api/app.py --model lstm
```

API接口：
- `POST /api/predict` — 手语识别（接收 base64 图片或 171 维特征）
- `POST /api/detect` — 仅检测手部关键点（返回 171 维特征）
- `POST /api/load_model` — 加载/切换模型
- `GET /api/health` — 健康检查
- `GET /api/models` — 模型列表
- `POST /api/collect` — 启动数据采集
- `POST /api/preprocess` — 启动数据预处理
- `POST /api/train` — 启动模型训练
- `GET /api/data/stats` — 数据统计
- `WS /ws/detect` — WebSocket 实时检测+预测

### 7. 全流程控制台

启动API服务后，访问 http://localhost:5000/dashboard 进入全流程控制台。

**控制台功能**：
| 模块 | 功能 |
|------|------|
| 📊 系统概览 | 数据统计、模型状态、工作流程 |
| 📹 数据采集 | 摄像头录制、视频导入、采集进度 |
| ⚙️ 数据预处理 | 配置参数、启动预处理 |
| 🎯 模型训练 | 选择模型、配置参数、训练监控 |
| 📈 模型评估 | 准确率、F1、混淆矩阵 |
| 📦 模型库 | 模型管理、加载/切换 |
| 🚀 实时推理 | 启动推理服务 |
| 🔌 API管理 | 服务状态、端点列表 |

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
├── 相对坐标：63维（21点 × 3坐标，相对于手腕）
├── 手指长度：4维（食指、中指、无名指、小指）
└── 关节角度：4维（4个手指根部角度）
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

### Q1: MediaPipe模型文件缺失

**错误**：`FileNotFoundError: 模型文件不存在`

**原因**：新版Task API需要本地模型文件。

**解决方案**：
```bash
python learning/download_models.py
```

### Q2: MediaPipe版本不兼容

**错误**：`AttributeError: module 'mediapipe' has no attribute 'tasks'`

**原因**：安装了旧版MediaPipe（<0.10.33），不支持新版Task API。

**解决方案**：
```bash
pip install --upgrade mediapipe>=0.10.33
```

### Q3: 旧版API报错

**错误**：`AttributeError: module 'mediapipe' has no attribute 'solutions'`

**原因**：新版MediaPipe（>=0.10.33）移除了旧版`solutions`模块。

**解决方案**：项目已更新为使用新版Task API，请确保安装正确版本：
```bash
pip install mediapipe>=0.10.33
```

### Q4: 模块导入失败

**错误**：`ModuleNotFoundError: No module named 'src'`

**解决方案**：工具脚本已自动添加项目路径，无需手动配置。

### Q5: 中文显示乱码

**现象**：界面显示"???"而非中文

**解决方案**：系统已自动加载Windows中文字体（微软雅黑/黑体），无需额外配置。

### Q6: 摄像头无法打开

**检查项**：
1. 摄像头是否被其他程序占用
2. 摄像头驱动是否正常
3. OpenCV是否正确安装：`pip install opencv-python`

### Q6: 升级后环境配置

**如果之前安装过旧版环境**，建议重建环境：
```bash
conda env remove -n lingxin -y
conda env create -f environment.yml
conda activate lingxin
```

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
| [📖 文档中心](docs/README.md) | 文档导航、推荐学习路径 |
| [01-快速入门](docs/01-quickstart.md) | 项目简介、环境搭建、5分钟体验 |
| [02-架构设计](docs/02-architecture.md) | 系统架构、模块设计、技术选型 |
| [03-数据采集指南](docs/03-data-collection.md) | 摄像头与视频批量采集全流程 |
| [04-模型训练](docs/04-training.md) | 训练、评估、推理完整流程 |
| [05-核心代码导读](docs/05-code-guide.md) | 源码结构、模块详解、扩展指南 |
| [06-设计模式实践](docs/06-design-patterns.md) | 模板方法、字典映射、架构实践 |
| [07-常见问题](docs/07-faq.md) | 环境/数据/模型问题排查 |
| [08-测试指南](docs/08-testing.md) | pytest使用、测试编写、最佳实践 |
| [🎓 新手学习教程](learning/README.md) | 10阶段零基础学习路线 |
| [🌐 Web演示](http://localhost:5000/) | 浏览器端在线手语识别 |
| [开发历程](docs/journal/development.md) | 项目开发大事记 |
| [实验记录](docs/journal/experiments.md) | 模型性能对比与调优 |

## 更新日志

### v0.3.0 (2026-06-21) — 第三个稳定版本
- 前端重构：统一导航组件 nav.js，所有页面链接统一为路由格式
- 文件重命名：docs-viewer.html → docs.html，删除冗余 index_new.html
- 新增页面：全流程控制台 dashboard.html
- 实时推理优化：移除骨架绘制，降低延迟，优化帧率(5FPS)
- 后端增强：API启动输出访问链接，兼容旧路由
- 依赖同步：pyproject.toml/requirements.txt/environment.yml 统一
- 文档更新：配置文件学习指南 09-configuration-guide.md
- 测试补充：LSTM/Trainer/Transformer 模型测试

### v0.2.0 (2026-06-16) — 第二个稳定版本
- 项目工程化：pyproject.toml、pre-commit hooks、GPU环境配置
- 文档全面同步：README/架构/开发记录/实验记录对齐项目实际
- learning教程：修复3个Bug，修正维度描述，新增配置管理/训练流程/数据增强3个教程（共13阶段）
- Web演示：新增 demo.html 在线手语识别和 MediaPipe WASM 资源
- 代码改进：API WebSocket支持，姿态降频优化，预划分数据集训练
- 测试：新增数据采集和配置管理测试

### v0.1.x — 第一个稳定版本
- **2026-06-11**: 新增测试指南文档（08-testing.md）
- **2026-06-10**: 文档结构重组，新增7篇学习路径文档
- **2026-06-01**: 升级MediaPipe至最新稳定版（>=0.10.33），迁移至Task API
- **2026-05-15**: 完善README文档，添加conda环境配置文件
- **2026-05-14**: 完成代码注释完善
- **2026-05-13**: 应用设计模式（模板方法模式），优化代码结构
- **2026-05-10**: 完成数据采集工具重写，支持时序数据录制

## 贡献指南

欢迎提交Issue和Pull Request！

## 开源协议

MIT License
