# 聆心（Lingxin）项目发展记录

## 项目简介

聆心（聆心/灵心）是一个基于深度学习的实时手语识别系统，聆听心灵的声音，搭建无声与有声世界的桥梁。

## 发展历程

### 2026年4月19日 - 项目初始化

#### 里程碑：项目结构搭建

完成了聆心手语识别系统的基础项目结构搭建，包括以下目录和文件：

**目录结构：**
```
Lingxin/
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据（视频/关键点）
│   │   ├── collected/           # 自采数据
│   │   └── ms_asl/              # MS-ASL数据集
│   └── processed/               # 预处理后的数据
│       └── csl_isolated/        # 自采孤立词数据
├── src/                         # 核心代码
│   ├── detection/               # 手部检测
│   ├── features/                # 特征工程
│   ├── models/                  # 模型定义
│   ├── training/                # 训练逻辑
│   └── utils/                   # 工具函数
├── tools/                       # 工具脚本
├── api/                         # API服务
├── docs/                        # 文档
├── tests/                       # 测试
├── requirements.txt
├── .gitignore
└── README.md
```

**核心模块：**
- `src/detection/hand_detector.py` - 基于MediaPipe的手部检测
- `src/features/feature_extractor.py` - 关键点到特征的转换
- `src/models/classifiers.py` - SVM/RF/MLP分类器
- `src/models/lstm_model.py` - LSTM模型
- `src/models/transformer_model.py` - Transformer模型
- `src/training/trainer.py` - 统一训练接口
- `src/utils/data_loader.py` - 数据加载
- `src/utils/visualization.py` - 可视化工具
- `src/utils/metrics.py` - 评估指标

**工具脚本：**
- `tools/collect_data.py` - 数据采集工具
- `tools/preprocess.py` - 数据预处理
- `tools/train.py` - 训练入口
- `tools/evaluate.py` - 评估入口
- `tools/inference.py` - 推理/演示

**初始提交：**
- Git仓库初始化
- 首次提交：`初始化项目结构`

---

### 2026年4月19日 - 框架迁移至PyTorch

#### 里程碑：深度学习框架切换

将项目从TensorFlow迁移至PyTorch框架，具体变更包括：

**依赖更新（requirements.txt）：**
- 移除TensorFlow依赖
- 添加PyTorch相关依赖：
  - torch>=2.0.0
  - torchvision>=0.15.0
- 为每个依赖包添加中文注释说明用途

**模型实现更新：**
- `src/models/lstm_model.py` - 从TensorFlow实现改为PyTorch实现
- `src/models/transformer_model.py` - 从TensorFlow实现改为PyTorch实现
- `src/training/trainer.py` - 更新训练接口以适配PyTorch

**提交记录：**
- `更新依赖和模型实现，切换到PyTorch`

---

### 2026年4月19日 - 环境配置

#### 里程碑：开发环境就绪

完成了项目开发环境的配置：

**虚拟环境：**
- 使用miniconda虚拟环境：`LingXin`
- Python版本：3.12.9

**依赖安装：**
- 已安装所有项目依赖
- PyTorch版本：2.11.0+cpu
- 验证CUDA可用性：False（当前为CPU版本）

**文档更新：**
- 更新README.md，添加详细的依赖说明表格
- 添加安装方法和虚拟环境激活说明

**提交记录：**
- `更新文档，添加详细的依赖说明`

---

### 2026年4月19日 - 上半身姿态检测功能

#### 里程碑：关键点检测扩展

在原有的手部检测基础上，增加了上半身姿态检测功能：

**新增模块（src/detection/hand_detector.py）：**

1. **HandDetector类**（原有，保持不变）
   - 使用MediaPipe Hands检测手部21个关键点

2. **PoseDetector类**（新增）
   - 使用MediaPipe Pose检测上半身姿态
   - 只提取上半身15个关键点（索引0~14）
   - 每个关键点包含x, y, z坐标

3. **HolisticDetector类**（新增）
   - 组合HandDetector + PoseDetector
   - 一次调用同时返回手部和姿态关键点
   - 返回格式：numpy array，形状为(171,)
     - 左手：63维（21×3）
     - 右手：63维（21×3）
     - 姿态：45维（15×3）

**提交记录：**
- `增加PoseDetector和HolisticDetector类，支持上半身姿态检测`

---

### 2026年4月19日 - 词汇表创建

#### 里程碑：数据基础建立

创建了项目词汇表，为后续数据采集和模型训练奠定基础：

**词汇表（data/vocab.csv）：**
- 共51个常用手语词汇
- 涵盖7大类别：
  - 问候（8个）：你好、谢谢、对不起、没关系、再见、你好吗、认识你、欢迎
  - 人称（6个）：我、你、他、她、我们、他们
  - 数字（10个）：一至十
  - 家庭（6个）：爸爸、妈妈、哥哥、姐姐、弟弟、妹妹
  - 动作（8个）：吃、喝、睡、走、跑、坐、站、看
  - 形容词（6个）：好、坏、大、小、多、少
  - 场景（6个）：学校、医院、家、工作、帮助、喜欢

**提交记录：**
- `添加手语词汇表vocab.csv，包含51个常用词汇`

---

## 技术栈

| 类别 | 技术 | 版本要求 |
|------|------|----------|
| 深度学习框架 | PyTorch | >=2.0.0 |
| 手部检测 | MediaPipe | >=0.10.0 |
| 图像处理 | OpenCV | >=4.8.0 |
| 机器学习 | scikit-learn | >=1.3.0 |
| API服务 | Flask | >=3.0.0 |

## 远程仓库

- GitHub仓库：git@github.com:readant/Lingxin.git

## 下一步计划

- [ ] 数据采集工具完善
- [ ] 特征工程模块优化
- [ ] 模型训练流程验证
- [ ] 实时推理功能测试
- [ ] API服务部署