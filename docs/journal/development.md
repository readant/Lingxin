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

### 4.8 MediaPipe 升级至Task API（2026年6月）

**里程碑**：升级关键点检测引擎

**变更内容**：
- MediaPipe 从 0.10.5 升级到 **>=0.10.33**（最新稳定版）
- 迁移至新版 Task API（`mp.tasks.vision`），替代旧版 `mp.solutions`
- 新增模型文件下载流程（`learning/download_models.py`）
- 更新 README 安装说明，推荐 conda 环境

**升级优势**：
- 推理速度优化
- 支持 WebGPU 部署
- 内存占用优化
- 现代 API 设计

### 4.9 项目工程化重构（2026年6月）

**里程碑**：提升代码工程化水平

**新增模块**：
1. **`src/config.py`** — 统一配置管理
   - 使用 Python dataclass 实现 `ProjectConfig` 单例
   - 集中管理路径、超参数、模型配置
   - 消除各模块的硬编码

2. **`src/constants.py`** — 共享常量
   - 手部/姿态骨架连接定义
   - 特征维度常量
   - 序列参数常量

3. **`src/utils/logger.py`** — 统一日志系统
   - 替代散落的 `print()` 调用
   - 支持控制台和文件输出
   - 标准化的日志格式

4. **`src/features/augmentation.py`** — 数据增强模块
   - 随机平移、缩放、噪声注入
   - 时间扭曲（模拟动作速度变化）
   - 随机关键点遮挡

### 4.10 CSL 公开数据集集成（2026年6月）

**里程碑**：引入大规模公开数据集

**新增内容**：
- `data/csl_vocab.csv` — CSL 数据集词汇表（~1058个词汇）
- `data/videos/CSL_basic_dataset/` — CSL 公开数据集视频
- `tools/collect_from_video.py` — 视频批量采集工具

**采集工具**：
- 支持三种文件名格式自动解析（词汇/人员/序号）
- 跳帧采样（`--frame-skip`）控制等效帧率
- 置信度过滤（`--min-confidence`）保证质量
- 输出与摄像头采集格式完全一致，可混合使用

### 4.11 预处理重构 — 按人员划分数据集（2026年6月）

**里程碑**：防止数据泄露，科学划分数据集

**核心改进**：
- 支持按人员划分训练/验证/测试集（`--split-by-person`）
- 自动解析文件名中的人员ID
- 保证同一人的数据不会跨集合出现
- 新增 `--summary` 模式查看数据人员分布
- 输出 `split_info.json` 记录划分详情

## 五、当前项目状态

### 5.1 已完成功能

| 模块 | 状态 | 说明 |
|------|------|------|
| 手部检测 | ✅ 完成 | MediaPipe 手部关键点检测（Task API） |
| 姿态检测 | ✅ 完成 | MediaPipe 上半身姿态检测（Task API） |
| 摄像头采集 | ✅ 完成 | 支持时序录制、按人员管理、实时预览 |
| 视频批量采集 | ✅ 完成 | 从视频文件自动提取关键点序列 |
| 配置管理 | ✅ 完成 | 统一 config.py + constants.py |
| 日志系统 | ✅ 完成 | logging 模块替代 print |
| 数据增强 | ✅ 完成 | 平移、缩放、噪声、时间扭曲、遮挡 |
| CSL数据集 | ✅ 完成 | ~1058个词汇的公开数据集集成 |
| 数据预处理 | ✅ 完成 | 标准化、序列对齐、按人员划分 |
| 传统ML模型 | ✅ 完成 | SVM、随机森林、MLP |
| 深度学习模型 | ✅ 完成 | LSTM、Transformer |
| 模型训练 | ✅ 完成 | 统一训练接口，预划分数据自动检测 |
| 模型评估 | ✅ 完成 | 多指标评估和混淆矩阵 |
| 实时推理 | ✅ 完成 | 摄像头实时识别 |

### 5.2 当前工作重点

| 任务 | 状态 | 说明 |
|------|------|------|
| 数据积累 | 🟡 进行中 | 持续扩充 CSL 及自采数据集 |
| 模型性能提升 | 🟡 进行中 | 基于更大数据集重新训练和调优 |

### 5.3 待开发功能

| 功能 | 优先级 | 说明 |
|------|--------|------|
| API服务完善 | 高 | Flask RESTful API 增强 |
| 模型部署优化 | 中 | ONNX/TensorRT 模型转换 |
| Web演示界面 | 中 | 可视化演示界面 |
| 连续手语识别 | 低 | 支持连续手语（非孤立词） |

## 六、下一步计划

### 6.1 短期目标（1-2周）
1. 验证 CSL 数据集在大模型上的训练效果
2. 完善 API 服务，支持线上预测
3. 补充实验数据，更新模型性能对比

### 6.2 中期目标（1-2月）
1. 模型超参数调优（网格搜索/贝叶斯优化）
2. 模型压缩与加速（ONNX导出）
3. 扩充自采数据，增强数据多样性

### 6.3 长期目标（3-6月）
1. 部署到生产环境
2. 支持连续手语识别
3. 开发移动端/Web端应用

---

> **作者提示**：本项目遵循敏捷开发原则，持续迭代改进。欢迎提交Issue和Pull Request！
