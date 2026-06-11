# 01 — 快速入门

> **学习目标**：了解项目背景，完成环境搭建，跑通最小闭环。

## 一、项目简介

**聆心（Lingxin）** 是一个基于深度学习的实时手语识别系统，旨在搭建无声与有声世界的桥梁。

- **核心能力**：识别中国手语（CSL）孤立词汇
- **技术栈**：PyTorch + MediaPipe (Task API) + scikit-learn
- **支持模型**：SVM、随机森林、MLP、LSTM、Transformer
- **数据规模**：自建50词 + CSL公开数据集 ~1058词

### 手语识别流程

```
摄像头/视频 → MediaPipe关键点检测 → 特征提取(71维) → 模型预测 → 识别结果
                    ↓                       ↓              ↓
              171维原始向量            特征向量         手语词汇
```

### 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 手部关键点检测 | ✅ | MediaPipe 21点 + 姿态15点 |
| 数据采集 | ✅ | 摄像头实时 + 视频批量 |
| 特征提取 | ✅ | 相对坐标、手指长度、关节角度 |
| 模型训练 | ✅ | 5种模型，按人员划分数据集 |
| 模型评估 | ✅ | 多指标 + 混淆矩阵 |
| 实时推理 | ✅ | 摄像头实时识别 |

## 二、环境搭建

### 推荐方式：conda 环境文件

```bash
# 1. 克隆项目
git clone git@github.com:readant/Lingxin.git
cd Lingxin

# 2. 创建 conda 环境
conda env create -f environment.yml

# 3. 激活环境
conda activate lingxin

# 4. 验证安装
python -c "import torch; import mediapipe; print('安装成功')"
```

### 手动创建（如果 environment.yml 不可用）

```bash
conda create -n lingxin python=3.10 -y
conda activate lingxin
conda install "pytorch>=2.0.0,<2.12.0" torchvision cpuonly -c pytorch -y
conda install "scikit-learn>=1.3.0,<1.5.0" pandas "numpy>=1.24.0" matplotlib seaborn tqdm pillow opencv flask -y
pip install "mediapipe>=0.10.33" flask-cors
```

### 下载 MediaPipe 模型文件

```bash
# 首次使用前必须下载（新版 Task API 需要本地模型文件）
python learning/download_models.py
```

模型文件会保存到 `models/` 目录。

### 验证安装

```python
import sys
print(f"Python: {sys.version}")

import numpy; print(f"NumPy: {numpy.__version__}")
import cv2;   print(f"OpenCV: {cv2.__version__}")
import mediapipe; print("MediaPipe: OK")
import torch; print(f"PyTorch: {torch.__version__}")
import sklearn; print(f"scikit-learn: {sklearn.__version__}")
```

### 关键依赖版本

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.9–3.11 | 不支持 3.12+ |
| MediaPipe | **>=0.10.33** | 新版 Task API（必选） |
| PyTorch | >=2.0.0, <2.12.0 | 深度学习框架 |
| OpenCV | >=4.8.0 | 图像处理 |
| scikit-learn | >=1.3.0, <1.5.0 | 传统 ML 模型 |

## 三、5 分钟快速体验

### 第 1 步：数据采集（摄像头）

```bash
conda activate lingxin
python tools/collect_data.py
```

按提示输入人员ID，空格键开始/停止录制。

### 第 2 步：数据预处理

```bash
# 查看数据分布
python tools/preprocess.py --summary

# 按人员划分数据集
python tools/preprocess.py --split-by-person \
    --train-persons J L \
    --test-persons F
```

### 第 3 步：模型训练

```bash
# 快速验证（SVM）
python tools/train.py --model svm

# 深度学习模型
python tools/train.py --model lstm
```

### 第 4 步：实时推理

```bash
python tools/inference.py --model lstm --checkpoint models/lstm_model.pth
```

### 完整流程一览

```
数据采集         预处理            训练             推理
   │               │                │                │
   ▼               ▼                ▼                ▼
collect_data   preprocess      train.py        inference
   │               │                │                │
   ▼               ▼                ▼                ▼
.npy 文件      训练/测试集     模型文件         实时识别
```

## 四、支持模型一览

| 模型 | 类型 | 命令 | 适用场景 |
|------|------|------|----------|
| SVM | 传统ML | `--model svm` | 快速训练、实时推理 |
| 随机森林 | 传统ML | `--model rf` | 特征重要性分析 |
| MLP | 传统ML | `--model mlp` | 中等复杂度任务 |
| LSTM | 深度学习 | `--model lstm` | 时序数据建模 |
| Transformer | 深度学习 | `--model transformer` | 高精度要求 |

## 五、下一步

- 开始采集数据 → [03-数据采集指南](03-data-collection.md)
- 了解系统架构 → [02-架构设计](02-architecture.md)
- 遇到问题？→ [07-常见问题](07-faq.md)

---

*最后更新：2026-06-10*
