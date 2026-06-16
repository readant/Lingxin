# 聆心项目 - 新手友好版学习指南

## 🎯 学习目标

帮助零基础新手逐步掌握手语识别项目所需的技术栈，重点解决：
- 环境配置和依赖安装
- 实时检测窗口创建
- 关键点检测原理
- 数据采集和处理
- 模型训练和评估

---

## 📚 学习路线（13个阶段）

### 第0阶段：环境准备
| 文件 | 学习目标 |
|------|----------|
| `00_env_setup.py` | 检查Python环境和依赖安装 |
| `download_models.py` | 下载MediaPipe模型文件 |

### 第1阶段：Python基础
| 文件 | 学习目标 |
|------|----------|
| `01_python_basics.py` | Python基础语法、函数、类 |

### 第2阶段：OpenCV基础
| 文件 | 学习目标 |
|------|----------|
| `02_opencv_basics.py` | 图像读取、显示、保存 |
| `02_opencv_video.py` | 摄像头访问和视频处理 |

### 第3阶段：MediaPipe入门
| 文件 | 学习目标 |
|------|----------|
| `03_mediapipe_intro.py` | MediaPipe框架介绍、Task API、模型文件 |

### 第4阶段：手部检测
| 文件 | 学习目标 |
|------|----------|
| `04_hand_detection_simple.py` | 基础手部关键点检测 |
| `04_hand_detection_draw.py` | 绘制关键点和骨架 |

### 第5阶段：姿态检测
| 文件 | 学习目标 |
|------|----------|
| `05_pose_detection.py` | 人体姿态关键点检测 |

### 第6阶段：NumPy数据处理
| 文件 | 学习目标 |
|------|----------|
| `06_numpy_intro.py` | NumPy数组基础 |
| `06_numpy_operations.py` | NumPy进阶和性能优化 |

### 第7阶段：特征工程
| 文件 | 学习目标 |
|------|----------|
| `07_feature_extraction.py` | 171维原始向量 vs 71维提取特征 |

### 第8阶段：机器学习
| 文件 | 学习目标 |
|------|----------|
| `08_svm_intro.py` | SVM分类器原理和使用 |

### 第9阶段：深度学习
| 文件 | 学习目标 |
|------|----------|
| `09_lstm_intro.py` | LSTM原理、项目LSTMModel、训练流程 |

### 第10阶段：数据采集
| 文件 | 学习目标 |
|------|----------|
| `10_data_collection.py` | 项目采集工具介绍、简化版演示 |

### 第11阶段：配置管理（新增）
| 文件 | 学习目标 |
|------|----------|
| `11_config_management.py` | config.py 统一配置、constants.py 共享常量 |

### 第12阶段：训练流程（新增）
| 文件 | 学习目标 |
|------|----------|
| `12_training_pipeline.py` | preprocess → train → evaluate 完整流程 |

### 第13阶段：数据增强（新增）
| 文件 | 学习目标 |
|------|----------|
| `13_data_augmentation.py` | KeypointAugmenter 5种增强策略 |

---

## 📅 学习计划

### 第1周：基础准备
```
Day 1-2: 00_env_setup.py → download_models.py → 01_python_basics.py
```

### 第2周：图像处理
```
Day 3-4: 02_opencv_basics.py → 02_opencv_video.py
Day 5-6: 03_mediapipe_intro.py
```

### 第3周：关键点检测
```
Day 7-8: 04_hand_detection_simple.py → 04_hand_detection_draw.py
Day 9-10: 05_pose_detection.py
```

### 第4周：数据处理
```
Day 11-12: 06_numpy_intro.py → 06_numpy_operations.py
Day 13-14: 07_feature_extraction.py
```

### 第5周：模型训练
```
Day 15-16: 08_svm_intro.py
Day 17-18: 09_lstm_intro.py
```

### 第6周：项目实战
```
Day 19-20: 10_data_collection.py → 11_config_management.py
Day 21-22: 12_training_pipeline.py → 13_data_augmentation.py
```

---

## 🚀 快速开始

```powershell
# 1. 激活conda环境
conda activate lingxin

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
├── 11_config_management.py      # 配置管理（新增）
├── 12_training_pipeline.py      # 训练流程（新增）
└── 13_data_augmentation.py      # 数据增强（新增）
```

---

## 💡 学习建议

1. **循序渐进**：按照阶段顺序学习，不要跳过
2. **动手实践**：每段代码都要自己运行一遍
3. **记录笔记**：遇到问题和解决方案及时记录
4. **调试技巧**：使用print()调试变量
5. **寻求帮助**：遇到困难可以查看文档或提问

---

**祝您学习顺利！**
