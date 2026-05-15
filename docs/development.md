# 聆心手语识别系统 - 开发记录

## 一、项目简介

聆心（Lingxin）是一个基于深度学习的实时手语识别系统，致力于搭建无声与有声世界的桥梁。本记录跟踪项目的开发历程、问题解决和技术演进。

## 二、技术栈

| 类别 | 技术 | 版本要求 | 用途 |
|------|------|----------|------|
| 深度学习框架 | PyTorch | >=2.0.0 | 模型训练和推理 |
| 关键点检测 | MediaPipe | 0.10.5 | 手部和姿态检测 |
| 图像处理 | OpenCV | >=4.8.0 | 视频处理和显示 |
| 图像处理 | Pillow | >=10.0.0 | 中文绘制 |
| 机器学习 | scikit-learn | >=1.3.0 | 传统ML模型 |
| API服务 | Flask | >=3.0.0 | RESTful API |
| 可视化 | Matplotlib | >=3.7.0 | 绘图和展示 |
| 可视化 | Seaborn | >=0.13.0 | 统计图表 |

## 三、问题归档

### 3.1 Python模块导入路径错误

**现象**：
```
ModuleNotFoundError: No module named 'src'
```

**原因**：
运行 `python tools/collect_data.py` 时，Python无法找到 `src` 模块，因为项目根目录未添加到Python路径。

**解决方案**：
在工具脚本开头添加路径处理：
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### 3.2 MediaPipe版本不兼容

**现象**：
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

**原因**：
- 新版本 MediaPipe（0.10.33+）使用新的 `mediapipe.tasks.python` API
- 本项目使用旧的 `mp.solutions.hands` API

**解决方案**：
降级到兼容版本：
```bash
pip install mediapipe==0.10.5
```

**验证方法**：
```python
import mediapipe as mp
print(mp.solutions.hands)  # 正常输出模块路径
```

### 3.3 OpenCV中文字体显示乱码

**现象**：
使用 `cv2.putText()` 绘制中文时，显示为"???"乱码。

**原因**：
OpenCV默认字体不支持中文字符。

**解决方案**：
使用PIL库绘制中文：
```python
from PIL import Image, ImageDraw, ImageFont

# 加载中文字体
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)

# OpenCV转PIL
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)

# 绘制中文
draw = ImageDraw.Draw(pil_img)
draw.text((10, 10), "你好", font=font, fill=(255, 255, 255))

# PIL转OpenCV
frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

### 3.4 模型训练过拟合

**现象**：
训练准确率很高，但验证准确率很低。

**解决方案**：
1. 增加训练数据量
2. 添加数据增强（随机裁剪、旋转等）
3. 使用正则化（Dropout、L2正则）
4. 减少模型复杂度

## 四、开发历程

### 4.1 项目初始化（2026年4月）

**里程碑**：搭建项目基础结构

完成内容：
- 创建项目目录结构
- 配置Git仓库
- 编写基础依赖清单
- 初始化README文档

**目录结构设计**：
```
Lingxin/
├── src/          # 核心代码
├── tools/        # 工具脚本
├── api/          # API服务
├── data/         # 数据存储
├── docs/         # 文档
└── tests/        # 测试代码
```

### 4.2 框架迁移至PyTorch（2026年4月）

**里程碑**：选择PyTorch作为深度学习框架

**变更内容**：
- 更新 `requirements.txt`，移除TensorFlow，添加PyTorch
- 重写 `src/models/lstm_model.py`（PyTorch实现）
- 重写 `src/models/transformer_model.py`（PyTorch实现）
- 更新 `src/training/trainer.py` 适配PyTorch

**选择理由**：
- PyTorch动态图更适合科研和调试
- 社区活跃，学习资源丰富
- 与Python生态无缝衔接

### 4.3 上半身姿态检测功能（2026年4月）

**里程碑**：扩展关键点检测能力

**新增内容**：
- `PoseDetector` 类：检测上半身15个关键点
- `HolisticDetector` 类：组合手部和姿态检测
- 171维特征向量：左手63 + 右手63 + 姿态45

**设计考虑**：
- 单独的手部信息可能不足以区分某些手语
- 添加姿态信息可以提高识别准确率
- 保持向后兼容，原有手部检测功能不变

### 4.4 词汇表创建（2026年4月）

**里程碑**：建立数据采集基础

**词汇表设计**：
- 共50个常用词汇
- 涵盖7大类别（问候、人称、数字、家庭、动作、形容词、场景）
- 使用CSV格式存储，便于扩展

**文件格式**：`data/vocab.csv`
```csv
category,word,description
问候,你好,打招呼用语
问候,谢谢,表达感谢
...
```

### 4.5 数据采集工具重写（2026年4月）

**里程碑**：支持时序数据录制

**核心改进**：
1. 使用 `HolisticDetector` 获取171维特征
2. 空格键控制录制开始/停止
3. 实时显示录制帧计数
4. 自动质量检查（帧长度过滤）
5. 支持中文字体显示

**界面功能**：
- 实时显示当前词汇和采集进度
- 录制状态可视化（红色边框）
- 检测失败警告（黄色提示）

### 4.6 设计模式应用（2026年5月）

**里程碑**：代码质量提升

**应用的设计模式**：
1. **模板方法模式**：`BaseModel` 定义模型骨架
2. **字典映射替代策略模式**：消除 if-elif 分支
3. **依赖注入**：训练器与模型解耦

**优化效果**：
- 代码复用率提高
- 新增模型更加便捷
- 代码可读性提升

### 4.7 代码注释完善（2026年5月）

**里程碑**：提升代码可维护性

**注释覆盖范围**：
- `src/detection/hand_detector.py` ✓
- `src/features/feature_extractor.py` ✓
- `src/models/*.py` ✓
- `src/training/trainer.py` ✓
- `src/utils/*.py` ✓
- `tools/*.py` ✓

**注释规范**：
- 文件头注释：说明文件功能和使用方法
- 类注释：说明类的职责和设计意图
- 方法注释：包含参数说明和返回值
- 关键代码注释：解释复杂逻辑

## 五、当前项目状态

### 5.1 已完成功能

| 模块 | 状态 | 说明 |
|------|------|------|
| 手部检测 | ✅ 完成 | MediaPipe手部关键点检测 |
| 姿态检测 | ✅ 完成 | MediaPipe上半身姿态检测 |
| 数据采集 | ✅ 完成 | 支持时序录制和质量检查 |
| 数据预处理 | ✅ 完成 | 标准化、序列对齐、划分 |
| 传统ML模型 | ✅ 完成 | SVM、随机森林、MLP |
| 深度学习模型 | ✅ 完成 | LSTM、Transformer |
| 模型训练 | ✅ 完成 | 统一训练接口 |
| 模型评估 | ✅ 完成 | 多指标评估和混淆矩阵 |
| 实时推理 | ✅ 完成 | 摄像头实时识别 |

### 5.2 待开发功能

| 功能 | 优先级 | 说明 |
|------|--------|------|
| API服务 | 高 | Flask RESTful API |
| 模型部署 | 中 | ONNX/TensorRT优化 |
| 数据增强 | 中 | 提高模型泛化能力 |
| Web界面 | 低 | 可视化演示界面 |

## 六、下一步计划

### 6.1 短期目标（1-2周）
1. 完善API服务（`api/app.py`）
2. 测试实时推理功能
3. 整理项目文档

### 6.2 中期目标（1-2月）
1. 采集更多训练数据
2. 优化模型性能
3. 支持更多词汇

### 6.3 长期目标（3-6月）
1. 部署到生产环境
2. 支持连续手语识别
3. 开发移动端应用

---

> **作者提示**：本项目遵循敏捷开发原则，持续迭代改进。欢迎提交Issue和Pull Request！