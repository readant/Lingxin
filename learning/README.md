# 聆心项目 - 新手友好版学习指南

## 🎯 学习目标

帮助零基础新手逐步掌握手语识别项目所需的技术栈，重点解决：
- 环境配置和依赖安装
- 实时检测窗口创建
- 关键点检测原理
- 数据采集和处理
- 模型训练和评估
- 实时推理与 Web 部署

> 学习是为了适应机器学习和机器视觉项目：路线由通用基础快速收敛到 ML/MV 主线，
> 每个脚本都直接对接 `src/` 与 `tools/` 的真实实现。配合阅读下方"对应文档"，
> 脚本与文档互相印证，理解更完整。

---

## 📚 学习路线（15个阶段）

### 第0阶段：环境准备
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `00_env_setup.py` | 检查Python环境和依赖安装 | — |
| `download_models.py` | 下载MediaPipe模型文件 | — |

### 第1阶段：Python基础
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `01_python_basics.py` | Python基础语法、函数、类 | — |

### 第2阶段：OpenCV基础
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `02_opencv_basics.py` | 图像读取、显示、保存 | — |
| `02_opencv_video.py` | 摄像头访问和视频处理 | — |

### 第3阶段：MediaPipe入门
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `03_mediapipe_intro.py` | MediaPipe框架介绍、Task API、模型文件 | — |

### 第4阶段：手部检测
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `04_hand_detection_simple.py` | 基础手部关键点检测 | — |
| `04_hand_detection_draw.py` | 绘制关键点和骨架 | — |

### 第5阶段：姿态检测
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `05_pose_detection.py` | 人体姿态关键点检测 | — |

### 第6阶段：NumPy数据处理
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `06_numpy_intro.py` | NumPy数组基础 | — |
| `06_numpy_operations.py` | NumPy进阶和性能优化 | — |

### 第7阶段：特征工程
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `07_feature_extraction.py` | 171维原始向量 vs 71维提取特征 | `docs/design/05-code-guide.md` |

### 第8阶段：机器学习
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `08_svm_intro.py` | SVM分类器原理和使用 | — |

### 第9阶段：深度学习
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `09_lstm_intro.py` | LSTM原理、项目LSTMModel、训练流程 | — |

### 第10阶段：数据采集
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `10_data_collection.py` | 项目采集工具介绍、简化版演示 | `docs/usage/03-data-collection.md` |

### 第11阶段：配置管理
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `11_config_management.py` | config.py 统一配置、constants.py 共享常量 | `docs/dev/09-configuration-guide.md` |

### 第12阶段：训练流程
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `12_training_pipeline.py` | preprocess → train → evaluate 完整流程 | `docs/usage/04-training.md` |

### 第13阶段：数据增强
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `13_data_augmentation.py` | KeypointAugmenter 5种增强策略 | `docs/design/06-design-patterns.md` |

### 第14阶段：实时推理
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `14_inference.py` | 实时推理闭环，对接 tools/inference.py | `docs/usage/04-training.md` |

### 第15阶段：Web 部署
| 文件 | 学习目标 | 对应文档 |
|------|----------|----------|
| `15_api_intro.py` | Flask API、REST/WebSocket 实时识别 | `docs/usage/04-training.md` |

---

## 🚀 快速开始

```powershell
# 1. 激活conda环境
conda activate lingxin-gpu

# 2. 下载模型（首次运行）
python learning/download_models.py

# 3. 按顺序运行学习脚本
python learning/00_env_setup.py
python learning/01_python_basics.py
python learning/02_opencv_basics.py
python learning/02_opencv_video.py
python learning/03_mediapipe_intro.py
python learning/04_hand_detection_simple.py
python learning/04_hand_detection_draw.py
python learning/05_pose_detection.py
python learning/06_numpy_intro.py
python learning/06_numpy_operations.py
python learning/07_feature_extraction.py
python learning/08_svm_intro.py
python learning/09_lstm_intro.py
python learning/10_data_collection.py
python learning/11_config_management.py
python learning/12_training_pipeline.py
python learning/13_data_augmentation.py
python learning/14_inference.py
python learning/15_api_intro.py
```

---

## 📁 学习文件夹结构

```
learning/
├── README.md                     # 学习指南
├── download_models.py            # 模型下载脚本
├── 00_env_setup.py              # 环境检查
├── 01_python_basics.py          # Python基础
├── 02_opencv_basics.py          # OpenCV图像基础
├── 02_opencv_video.py           # OpenCV视频处理
├── 03_mediapipe_intro.py        # MediaPipe介绍
├── 04_hand_detection_simple.py  # 手部检测基础
├── 04_hand_detection_draw.py    # 绘制关键点
├── 05_pose_detection.py         # 姿态检测
├── 06_numpy_intro.py            # NumPy基础
├── 06_numpy_operations.py       # NumPy进阶
├── 07_feature_extraction.py     # 特征工程（171维 vs 71维）
├── 08_svm_intro.py              # SVM入门
├── 09_lstm_intro.py             # LSTM入门（引用项目LSTMModel）
├── 10_data_collection.py        # 数据采集实战
├── 11_config_management.py      # 配置管理
├── 12_training_pipeline.py      # 训练流程
├── 13_data_augmentation.py      # 数据增强
├── 14_inference.py              # 实时推理闭环
└── 15_api_intro.py              # Web 部署（Flask API）
```

---

## 💡 学习建议

1. **循序渐进**：按照阶段顺序学习，不要跳过
2. **动手实践**：每段代码都要自己运行一遍
3. **配套阅读**：学到第 7 阶段及以后，同步阅读每节"对应文档"中的 docs 章节，脚本与文档互相印证
4. **记录笔记**：遇到问题和解决方案及时记录
5. **调试技巧**：使用print()调试变量
6. **寻求帮助**：遇到困难可以查看文档或提问

---

**祝您学习顺利！**
