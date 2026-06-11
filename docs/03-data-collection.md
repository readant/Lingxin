# 聆心手语识别系统 — 数据采集指南

## 一、采集工具概览

本项目提供两种数据采集方式，互补使用：

| 工具 | 输入源 | 控制方式 | 适用场景 |
|------|--------|----------|----------|
| [collect_data.py](../tools/collect_data.py) | 实时摄像头 | 手动交互（空格录制） | 人工演示、实验室环境 |
| [collect_from_video.py](../tools/collect_from_video.py) | 视频文件 | 自动批处理 | 网络视频、已有素材批量入库 |

**共同输出**：两种工具产生的 `.npy` 文件格式完全一致，可混合使用。

---

## 二、数据目录与命名规范

### 2.1 目录结构

```
data/
├── vocab.csv                           # 自建词汇表（50个词）
├── csl_vocab.csv                       # CSL数据集词汇表（~1058个词）
├── videos/                             # 待处理的视频文件（不被 git 追踪）
│   ├── 你好_A_001.mp4                  # 自录视频（自由命名）
│   ├── 谢谢_B_001.mp4
│   └── CSL_basic_dataset/              # CSL公开数据集视频
│       ├── 你好.mp4
│       ├── 谢谢.mp4
│       └── ...
└── raw/
    └── collected/                      # 采集输出目录
        ├── 你好/
        │   ├── A_001.npy               # A 录制的第 1 条
        │   ├── A_001_meta.json         #   对应元信息
        │   ├── J_001.npy               # J 录制的第 1 条（视频采集）
        │   ├── J_001_meta.json
        │   └── ...
        ├── 谢谢/
        │   └── ...
        ├── 三明治/                     # CSL数据集词汇
        │   └── CSL_001.npy             # CSL表示来源为CSL数据集
        └── ...
```

### 2.2 文件命名规范

```
{人员ID}_{序号:03d}.npy          →  A_001.npy, L_010.npy
{人员ID}_{序号:03d}_meta.json     →  A_001_meta.json
```

- **人员ID**：单个大写字母（如 `A`, `L`, `F`）或字符串（如 `user001`），用于区分不同采集者
- **序号**：自动递增，不会被覆盖

### 2.3 数据格式

每个 `.npy` 文件：`(帧数, 171)` 的 float64 数组。

```
171维 = 左手63维（21点×3坐标）
     + 右手63维（21点×3坐标）
     + 姿态45维（15点×3坐标）
```

### 2.4 元信息文件（`_meta.json`）

```json
{
  "person_id": "J",
  "word": "你好",
  "category": "问候",
  "num_frames": 37,
  "feature_dim": 171,
  "timestamp": "2026-06-03T14:22:19",
  "file_name": "J_002.npy",
  "source_video": "你好_J.mp4",       // 仅视频采集工具有此字段
  "source_fps": 30.0,                 // 仅视频采集工具
  "valid_ratio": 0.85                 // 仅视频采集工具（有效帧比例）
}
```

---

## 三、工具一：摄像头实时采集

### 3.1 启动

```bash
conda activate lingxin
python tools/collect_data.py
```

按提示输入人员ID，进入采集界面。

### 3.2 界面说明

| 位置 | 内容 | 说明 |
|------|------|------|
| 左上 | `词: 你好 (问候)` | 当前词汇及类别 |
| 左上第二行 | `进度: 1/50` | 词汇表位置 |
| 右上 | `已录: 2/30` | **当前人员**的已录制数 |
| 右上第二行 | `总计: 13` | 该词**所有人员**的样本总数（不同时显示） |
| 底部 | 操作提示 | 快捷键列表 |
| 边框 | 绿色=待机 / 红色=录制中 | 状态指示 |

### 3.3 操作流程

```
准备 → [空格] → 3秒倒计时 → 录制中... → [空格]停止 → 回顾界面
                                                          ├─ [空格] 保存
                                                          ├─ [R] 重录
                                                          ├─ [D] 回放预览
                                                          └─ [ESC] 取消
```

### 3.4 键盘控制

| 按键 | 功能 |
|------|------|
| 空格 | 开始录制（3秒倒计时）/ 停止录制 |
| N | 下一个词汇 |
| P | 上一个词汇 |
| R | 删除当前词汇最后一个样本（仅删除自己的） |
| Q | 显示统计后退出 |
| ESC | 直接退出 |

### 3.5 质量控制

| 检查项 | 阈值 | 行为 |
|--------|------|------|
| 最短序列 | 15帧 | 不足则拒绝保存 |
| 最长序列 | 150帧 | 超长则中心裁剪 |
| 手部丢失 | 连续帧无手 | 黄色警告提示 |

### 3.6 采集建议

- 动作**连贯流畅**，从起始位到结束位一气呵成
- 手部保持在**画面中央**，距离摄像头 1-2 米
- 光线**充足均匀**，背景简洁
- 每个词建议录制 **20-30 个样本**，覆盖不同角度
- **多人协作**采集能大幅提升模型泛化能力

---

## 四、工具二：视频文件批量采集

### 4.1 适用场景

- 从网络下载的手语教学视频批量提取关键点
- 用手机预先录制，再统一导入
- 扩充数据集时快速处理存量视频

### 4.2 视频文件命名

将视频放入 `data/videos/` 目录，按以下格式命名：

```
{词汇}_{人员ID}_{序号}.mp4       →  你好_A_001.mp4   （推荐，信息完整）
{词汇}_{人员ID}.mp4              →  你好_A.mp4
{词汇}.mp4                      →  你好.mp4          （需配合 --person-id）
```

支持的视频格式：`.mp4` `.avi` `.mov` `.mkv` `.webm`

### 4.3 使用方法

```bash
# 基本用法：处理 data/videos/ 下所有视频
python tools/collect_from_video.py -i data/videos/

# 指定人员ID（覆盖文件名推断）
python tools/collect_from_video.py -i data/videos/ -p K

# 跳帧采样（原始30fps视频，每2帧取1帧 → 等效15fps）
python tools/collect_from_video.py -i data/videos/ --frame-skip 2

# 提高检测置信度阈值（过滤低质量检测）
python tools/collect_from_video.py -i data/videos/ --min-confidence 0.7

# 指定输出目录
python tools/collect_from_video.py -i data/videos/ -o data/raw/collected
```

### 4.4 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i / --input-dir` | 必填 | 视频文件目录 |
| `-o / --output-dir` | `data/raw/collected` | 数据输出目录 |
| `-p / --person-id` | 无 | 文件名中未包含人员时使用 |
| `--frame-skip` | 1 | 跳帧间隔：N帧中取1帧 |
| `--min-confidence` | 0.5 | MediaPipe 检测最低置信度 |

### 4.5 处理流程

```
data/videos/你好_A.mp4
        │
        ▼
  cv2.VideoCapture 逐帧读取
        │
        ▼
  HolisticDetector 提取关键点（每帧）
        │
        ├─ 无手帧 → 跳过
        └─ 有手帧 → 加入序列
        │
        ▼
  保存: data/raw/collected/你好/A_001.npy
       + A_001_meta.json（含 source_video、valid_ratio）
```

### 4.6 `--frame-skip` 选择建议

| 原始视频 | 建议设置 | 等效帧率 | 适用 |
|----------|----------|----------|------|
| 30fps | `--frame-skip 2` | 15fps | 动作缓慢的手语 |
| 30fps | `--frame-skip 1` | 30fps | 快速动作 |
| 60fps | `--frame-skip 3` | 20fps | 高速摄像机 |

---

## 五、查看数据分布

在预处理前，先了解数据的人员构成和词汇分布：

```bash
python tools/preprocess.py --summary
```

输出示例（混合自采 + CSL数据集）：

```
============================================================
数据人员分布摘要
============================================================
  A: 50 samples (你好:10, 谢谢:10, 再见:10, ...)
  J: 2 samples (你好:2)
  L: 10 samples (你好:10)
  CSL: 120 samples (三明治:1, 上午:1, 书:1, ...)

总计: 4 人, 182 个样本
============================================================
```

> **提示**：CSL 数据集视频采集后，人员ID默认为 `CSL`（可在 `collect_from_video.py` 中用 `--person-id` 覆盖），便于在数据分布中区分数据来源。

---

## 六、从采集到训练 — 完整流程

```bash
# 第1步：采集数据
#   方式A：摄像头手动采集
python tools/collect_data.py

#   方式B：视频文件批量导入（自录视频）
python tools/collect_from_video.py -i data/videos/

#   方式C：CSL公开数据集导入
python tools/collect_from_video.py -i data/videos/CSL_basic_dataset/ -p CSL

# 第2步：查看数据分布（可选）
python tools/preprocess.py --summary

# 第3步：按人员划分预处理
#   --train-persons: 训练集人员（量多、来源广）
#   --test-persons:  测试集人员（训练集中未出现过的人）
python tools/preprocess.py --split-by-person \
    --train-persons J L CSL \
    --test-persons F

# 第4步：训练
python tools/train.py --model lstm
```

预处理输出：

```
data/processed/csl_isolated/
├── X_train.npy / y_train.npy      # 训练集（J + L 的数据）
├── X_test.npy / y_test.npy        # 测试集（F 的数据）
├── sequence_X_train.npy / ...     # 序列版本（深度学习用）
├── class_labels.npy               # 类别标签映射
└── split_info.json                # 划分详情（可追溯）
```

---

## 七、常见问题

### Q1: 检测不到手部

- 确保光线充足，背景简洁
- 手掌面向摄像头，五指张开
- 降低 `--min-confidence`（如 0.3）

### Q2: 录制的序列太短（<15帧）

- 动作放慢，确保录制时间 ≥ 1 秒
- 视频采集时检查 `--frame-skip` 是否过大

### Q3: 视频采集后有效帧太少

- 查看 `_meta.json` 中的 `valid_ratio` 字段
- 如果 < 0.5，说明视频中大部分时间手不在画面内
- 检查视频是否正确（是否有完整的手语动作）

### Q4: 为什么界面只显示我的样本数？

摄像头采集工具界面的 `已录: N/30` 只统计**当前登录人员**的样本。如果想知道所有人员的总量，看下面那行 `总计: N`，或运行 `--summary`。

### Q5: 多人采集的数据如何管理？

所有人员的数据混放在同一词汇目录下，通过文件名前缀区分。预处理时通过 `--split-by-person` 按人员划分，系统保证同一人的数据不会跨越训练/验证/测试集。

### Q6: 中文显示乱码（界面或终端）

界面：系统会自动查找 Windows 中文字体（微软雅黑 → 黑体 → 宋体）。如果全部缺失，请安装一种。

终端：在 Windows 终端运行前设置编码：

```bash
set PYTHONIOENCODING=utf-8
python tools/collect_from_video.py -i data/videos/
```

---

> **核心原则**：数据质量 > 数据数量。宁可少而精，不要多而滥。同一人的数据绝不出现在训练和测试两个集合中。
