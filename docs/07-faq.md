# 07 — 常见问题

> 遇到问题时，先查看以下分类。大部分问题都能在这里找到答案。

## 一、环境问题

### Q1: MediaPipe 模型文件缺失

**错误**：`FileNotFoundError: 模型文件不存在`

**原因**：新版 MediaPipe Task API 需要本地模型文件。

**解决**：
```bash
python learning/download_models.py
```

### Q2: MediaPipe 版本不兼容

**错误**：`AttributeError: module 'mediapipe' has no attribute 'tasks'`

**原因**：安装了旧版 MediaPipe（<0.10.33）。

**解决**：
```bash
pip install --upgrade "mediapipe>=0.10.33"
```

### Q3: 旧版 API 报错

**错误**：`AttributeError: module 'mediapipe' has no attribute 'solutions'`

**原因**：新版 MediaPipe（>=0.10.33）已移除旧版 `solutions` 模块。

**解决**：项目已迁移至 Task API，安装正确版本即可：
```bash
pip install "mediapipe>=0.10.33"
```

### Q4: 模块导入失败

**错误**：`ModuleNotFoundError: No module named 'src'`

**解决**：工具脚本已自动添加项目路径到 `sys.path`，确保在项目根目录运行：
```bash
cd /path/to/Lingxin
python tools/train.py --model svm
```

### Q5: Python 版本不兼容

**错误**：`ImportError: This platform does not support PyTorch`

**解决**：推荐使用 Python 3.9–3.11：
```bash
conda create -n lingxin python=3.10
conda activate lingxin
```

### Q6: PyTorch GPU 不可用

`torch.cuda.is_available()` 返回 `False`。

**排查步骤**：
```bash
nvidia-smi                      # 检查驱动
python -c "import torch; print(torch.version.cuda)"  # 检查 CUDA 版本
```

**解决**：卸载 CPU 版本，安装 GPU 版本：
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q7: 升级后环境异常

如果之前安装过旧版环境，建议重建：
```bash
conda env remove -n lingxin -y
conda env create -f environment.yml
conda activate lingxin
```

## 二、数据问题

### Q8: 数据加载失败

**错误**：`FileNotFoundError: No such file or directory: 'data/processed/X.npy'`

**解决**：先运行预处理生成数据文件：
```bash
python tools/preprocess.py
```

### Q9: 检测不到手部

- 确保光线充足，背景简洁
- 手掌面向摄像头，五指张开
- 降低置信度阈值：`--min-confidence 0.3`

### Q10: 录制的序列太短（<15帧）

- 动作放慢，确保录制时间 ≥ 1 秒
- 视频采集时检查 `--frame-skip` 是否过大

### Q11: 视频采集后有效帧太少

查看 `_meta.json` 中的 `valid_ratio` 字段：
- 如果 < 0.5，说明视频中大部分时间手不在画面内
- 检查视频是否正确（是否有完整的手语动作）

### Q12: 中文显示乱码

**界面**：系统会自动查找 Windows 中文字体（微软雅黑 → 黑体 → 宋体）。如果全部缺失，请安装一种。

**终端**：
```bash
set PYTHONIOENCODING=utf-8
python tools/collect_from_video.py -i data/videos/
```

## 三、模型问题

### Q13: 训练 loss 不下降 / 为 NaN

**排查步骤**：
```python
import numpy as np

# 1. 检查数据是否有 NaN
print(np.isnan(X).sum())

# 2. 检查数据范围
print(f"X min: {X.min()}, max: {X.max()}")

# 3. 尝试调小学习率：lr=0.0001
```

**常见原因**：

| 原因 | 解决方案 |
|------|----------|
| 学习率过大 | 调小学习率，如 0.0001 |
| 数据未标准化 | 使用 StandardScaler |
| 数据有 NaN | 数据清洗，填充或删除 |
| 类别不平衡 | 使用类别权重或过采样 |

### Q14: 模型预测准确率为 0 / 只输出一个类别

**排查**：
```python
y_pred = model.predict(X_test)
print("预测分布:", Counter(y_pred))
print("真实分布:", Counter(y_test))
```

**常见原因**：
1. 标签全为同一类别
2. 模型输出层维度与类别数不匹配
3. 训练轮次不够

### Q15: GPU 内存不足 (OOM)

**错误**：`RuntimeError: CUDA out of memory`

**解决**：
```python
# 减小 batch_size
batch_size = 16  # 从 32 改为 16

# 及时清理显存
torch.cuda.empty_cache()

# 减少 DataLoader workers
num_workers = 0
```

### Q16: 小数据集快速验证

```python
# 先在少量数据上训练，验证模型能过拟合
X_small = X[:100]
y_small = y[:100]
model.fit(X_small, y_small)
train_acc = model.score(X_small, y_small)
print(f"训练准确率: {train_acc:.4f}")  # 应该接近 100%
```

## 四、摄像头问题

### Q17: 摄像头无法打开

**检查项**：
1. 摄像头是否被其他程序占用（如微信、Zoom）
2. OpenCV 是否正确安装：`pip install opencv-python`
3. 尝试指定摄像头索引：修改 `config.camera_index` 或传 `--camera 1`

## 五、数据采集问题

### Q18: 为什么界面只显示我的样本数？

摄像头采集工具界面的 `已录: N/30` 只统计**当前登录人员**的样本。查看总量运行：
```bash
python tools/preprocess.py --summary
```

### Q19: 多人采集的数据如何管理？

所有人员的数据混放在同一词汇目录下，通过文件名前缀区分。预处理时通过 `--split-by-person` 按人员划分，系统保证同一人的数据不会跨集合。

### Q20: 数据采集器运行时卡死/无响应

**现象**：运行 `tools/collect_data.py` 后，程序偶尔会完全卡死，无法响应键盘操作，只能强制关闭。

**原因**：回顾界面（`_show_review`）和回放暂停（`_playback_sequence`）使用了 `cv2.waitKey(0)` 无限等待按键。当 OpenCV 窗口失去焦点时（如点击其他窗口），键盘事件无法送达，导致主线程永久阻塞。

**解决**：已修复为 `waitKey(33)` 循环 + 窗口关闭检测。如果仍遇到此问题，请确认代码为最新版本。

### Q21: 预览回放后主界面卡顿、帧率下降

**现象**：录制完成后按 `D` 预览回放序列，返回主采集界面后，摄像头画面明显卡顿，帧率比之前降低。

**原因**：

1. **Preview 窗口泄露**（主因）：`_playback_sequence` 通过 `cv2.imshow('Preview', frame)` 打开了第二个 OpenCV 窗口，但回放结束后未销毁。返回主循环后，OpenCV highgui 需要同时管理 `Preview` 和 `Data Collection` 两个窗口，`cv2.waitKey(1)` 分摊处理两个窗口的事件循环，导致帧率下降。
2. **摄像头缓冲区积压**：回放期间（最长约 5 秒）摄像头持续捕获帧，缓冲区积累大量旧帧，返回主循环后 `cap.read()` 拿到的是延迟帧。

**解决**：已在代码中修复（`tools/collect_data.py`）：

- `_playback_sequence` 使用 `try/finally` 确保 `cv2.destroyWindow('Preview')` 在回放结束后必定执行
- `_show_review` 在预览回放返回后立即调用 `_flush_capture` 清空摄像头积压帧

如果使用旧版本代码遇到此问题，请更新到最新版本。

## 六、自检清单

遇到问题时，按以下顺序排查：

```
□ 1. 环境检查
   □ Python 版本 3.9–3.11
   □ MediaPipe >= 0.10.33
   □ 所有依赖安装成功
   □ 模型文件已下载（models/*.task）

□ 2. 数据检查
   □ 数据文件存在
   □ 数据格式正确（.npy）
   □ X 和 y 样本数一致
   □ 没有 NaN 或异常值

□ 3. 模型检查
   □ 模型输出维度正确
   □ 损失函数选择正确
   □ 学习率设置合理

□ 4. 训练检查
   □ 训练 loss 在下降
   □ 验证准确率在提升
   □ 没有内存溢出
```

## 七、获取更多帮助

如果以上都无法解决，准备以下信息寻求帮助：

- Python 版本、PyTorch 版本、MediaPipe 版本
- 操作系统
- 完整的错误堆栈信息
- 已尝试的解决方法

---

*最后更新：2026-06-15*
