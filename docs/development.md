# 聆心（Lingxin）项目发展记录

## 项目简介

聆心（聆心/灵心）是一个基于深度学习的实时手语识别系统，聆听心灵的声音，搭建无声与有声世界的桥梁。

## 问题归档

### 2026年4月19日 - 问题记录

#### 问题1：Python模块导入路径错误

**现象：**
```
ModuleNotFoundError: No module named 'src'
```

**原因：**
运行 `python tools/collect_data.py` 时，Python无法找到 `src` 模块，因为项目根目录未添加到Python路径。

**解决方案：**
在 `tools/collect_data.py` 开头添加：
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

---

#### 问题2：mediapipe版本不兼容

**现象：**
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**原因：**
- 最初安装的 mediapipe 0.10.33 使用新的 `mediapipe.tasks.python` API
- 原代码使用的是旧的 `mp.solutions.hands` API

**解决方案：**
降级到 mediapipe 0.10.5：
```bash
pip install mediapipe==0.10.5
```

**验证：**
```python
import mediapipe as mp
print(mp.solutions.hands)  # 正常输出模块路径
```

---

#### 问题3：OpenCV中文字体显示为"???"

**现象：**
使用 `cv2.putText()` 绘制中文时，显示为"???"乱码。

**原因：**
OpenCV默认字体不支持中文字符。

**解决方案：**
使用PIL库绘制中文：
```python
from PIL import Image, ImageDraw, ImageFont

# 加载中文字体
def _init_font(self):
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            self.font = ImageFont.truetype(fp, 20)
            break

# OpenCV转PIL
def _cv2_to_pil(self, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# PIL转OpenCV
def _pil_to_cv2(self, pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 绘制中文
def _draw_text_pil(self, pil_img, text, position, font_size=20, color=(255, 255, 255)):
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=self.font, fill=color)
```

---

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
- 共50个常用手语词汇
- 涵盖7大类别：
  - 问候（8个）：你好、谢谢、对不起、没关系、再见、你好吗、认识你、欢迎
  - 人称（6个）：我、你、他、她、我们、他们
  - 数字（10个）：一至十
  - 家庭（6个）：爸爸、妈妈、哥哥、姐姐、弟弟、妹妹
  - 动作（8个）：吃、喝、睡、走、跑、坐、站、看
  - 形容词（6个）：好、坏、大、小、多、少
  - 场景（6个）：学校、医院、家、工作、帮助、喜欢

**提交记录：**
- `添加手语词汇表vocab.csv，包含50个常用词汇`

---

### 2026年4月19日 - 数据采集工具重写

#### 里程碑：支持时序数据录制

完全重写了 `tools/collect_data.py`，解决了单帧保存的问题：

**核心改进：**
1. 使用 `HolisticDetector` 获取171维关键点向量
2. 空格键控制录制开始/停止，持续采集时序数据
3. 录制时检测手部，黄色警告检测不到手的情况
4. 质量检查：序列<15帧不保存，>150帧截取中间部分

**界面元素：**
- 左上角：当前词汇 + 进度（X/50）
- 右上角：已录次数 / 目标30次
- 底部：操作提示
- 录制中：红色边框 + 帧计数

**键盘控制：**
- 空格：开始/停止录制
- N：下一个词
- P：上一个词
- Q：显示统计后退出
- ESC：直接退出

**提交记录：**
- `重写数据采集工具，支持时序录制和HolisticDetector`

---

## 技术栈

| 类别 | 技术 | 版本要求 |
|------|------|----------|
| 深度学习框架 | PyTorch | >=2.0.0 |
| 手部检测 | MediaPipe | 0.10.5 |
| 图像处理 | OpenCV | >=4.8.0 |
| 图像处理(PIL) | Pillow | >=10.0.0 |
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