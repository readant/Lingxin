# 04 — 模型训练与评估

> **学习目标**：掌握从数据预处理到模型训练、评估、推理的完整流程。

## 一、训练流程概览

```
原始数据(.npy) → 预处理(preprocess.py) → 训练(train.py) → 评估(evaluate.py) → 推理(inference.py)
     │                    │                      │                 │                  │
     ▼                    ▼                      ▼                 ▼                  ▼
 关键点序列         标准化+划分            模型文件(.pth)     准确率/混淆矩阵    实时识别结果
```

## 二、数据预处理

### 基本用法

```bash
# 查看数据分布
python tools/preprocess.py --summary

# 标准预处理（所有数据随机划分）
python tools/preprocess.py

# 按人员划分（推荐：防止数据泄露）
python tools/preprocess.py --split-by-person \
    --train-persons J L CSL \
    --test-persons F
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--input` | 原始数据目录（默认 `data/raw/collected`） |
| `--output` | 输出目录（默认 `data/processed/csl_isolated`） |
| `--split-by-person` | 启用按人员划分模式 |
| `--train-persons` | 训练集人员ID（空格分隔） |
| `--val-persons` | 验证集人员ID（可选） |
| `--test-persons` | 测试集人员ID |
| `--summary` | 只查看数据分布，不做预处理 |

### 输出文件

```
data/processed/csl_isolated/
├── X_train.npy / y_train.npy       # 训练集（普通特征，传统ML用）
├── X_val.npy / y_val.npy           # 验证集
├── X_test.npy / y_test.npy         # 测试集
├── sequence_X_train.npy / ...      # 序列版本（深度学习用）
├── class_labels.npy                # 类别标签映射
└── split_info.json                 # 划分详情（可追溯）
```

> **重要**：按人员划分保证同一人的数据不会跨集合出现，能更真实反映模型泛化能力。

## 三、模型训练

### 基本用法

```bash
# 训练传统 ML 模型
python tools/train.py --model svm
python tools/train.py --model rf
python tools/train.py --model mlp

# 训练深度学习模型
python tools/train.py --model lstm
python tools/train.py --model transformer

# 指定数据目录
python tools/train.py --model lstm --data data/processed/csl_isolated
```

### 工作机制

`train.py` 会自动检测数据目录：
- 如果存在 `X_train.npy` / `X_val.npy` / `X_test.npy` → 使用预划分数据
- 如果只有 `X.npy` / `y.npy` → 自动随机划分

深度学习模型额外检测 `sequence_X_train.npy` 等序列版本。

## 四、模型对比

| 模型 | 类型 | 输入格式 | 训练速度 | 推理速度 | 适用场景 |
|------|------|----------|----------|----------|----------|
| SVM | 传统ML | (samples, 71) | 快（秒级） | <1ms | 快速原型、小数据集 |
| 随机森林 | 传统ML | (samples, 71) | 快（秒级） | <1ms | 特征分析、抗噪 |
| MLP | 传统ML | (samples, 71) | 中（分钟） | <1ms | 中等规模 |
| LSTM | 深度学习 | (samples, 30, 71) | 慢（10分钟） | ~5ms | 时序建模 |
| Transformer | 深度学习 | (samples, 30, 71) | 慢（15分钟） | ~8ms | 高精度 |

### 模型选型建议

```
数据量 <1000 样本  → SVM 或 随机森林
数据量 1000-5000   → MLP 或 LSTM
数据量 >5000       → Transformer
需要实时推理        → SVM、RF、MLP（<1ms）
追求最高精度        → Transformer
```

## 五、模型评估

```bash
# 评估已训练的模型
python tools/evaluate.py --model lstm --checkpoint models/lstm_model.pth

# 指定测试数据
python tools/evaluate.py --model svm --checkpoint models/svm_model.pkl --data data/processed/csl_isolated
```

### 评估指标

| 指标 | 含义 | 关注场景 |
|------|------|----------|
| **准确率 (Accuracy)** | 整体正确预测比例 | 类别均衡时 |
| **精确率 (Precision)** | 预测为正例中真实正例的比例 | 减少误报 |
| **召回率 (Recall)** | 真实正例中被预测出的比例 | 减少漏报 |
| **F1 分数** | 精确率和召回率的调和平均 | 综合评估 |

## 六、实时推理

```bash
# 启动摄像头实时识别
python tools/inference.py --model lstm --checkpoint models/lstm_model.pth

# 指定摄像头
python tools/inference.py --model transformer --checkpoint models/transformer_model.pth --camera 1
```

**操作说明**：
- 摄像头实时显示检测画面和关键点
- 做出手语动作后，系统自动识别并输出结果
- 按 `Q` 键退出

## 七、API 服务

```bash
# 启动 Flask API
python api/app.py
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/predict` | 上传关键点数据，返回识别结果 |
| GET | `/models` | 获取可用模型列表 |
| POST | `/train` | 触发模型训练（开发中） |

## 八、完整工作流示例

```bash
# Step 1: 采集数据（摄像头）
python tools/collect_data.py

# Step 1b: 或从视频导入
python tools/collect_from_video.py -i data/videos/

# Step 2: 查看数据分布
python tools/preprocess.py --summary

# Step 3: 按人员划分预处理
python tools/preprocess.py --split-by-person \
    --train-persons J L --test-persons F

# Step 4: 训练模型
python tools/train.py --model lstm

# Step 5: 评估模型
python tools/evaluate.py --model lstm --checkpoint models/lstm_model.pth

# Step 6: 实时推理
python tools/inference.py --model lstm --checkpoint models/lstm_model.pth
```

## 九、下一步

- 深入代码实现 → [05-核心代码导读](05-code-guide.md)
- 理解架构设计 → [02-架构设计](02-architecture.md)
- 遇到问题？→ [07-常见问题](07-faq.md)

---

*最后更新：2026-06-10*
