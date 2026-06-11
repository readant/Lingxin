# 08 — 测试指南

> **学习目标**：理解测试的重要性，掌握 pytest 使用方法，学会为项目编写测试用例。

## 一、为什么需要测试

### 1.1 测试 vs Git 回退

很多初学者会问："我用 git 不是同样可以回退代码吗？为什么还要写测试？"

| | Git 回退 | 测试验证 |
|--|----------|----------|
| **作用** | 记录代码历史，可以撤销改动 | 验证代码行为是否正确 |
| **解决** | "我想回到上个版本" | "这行代码改完后，功能还对吗" |
| **能力** | 撤销代码变更 | 发现逻辑错误 |

**关键区别**：

```
你改了 trainer.py 的 loss 计算逻辑
↓
Git：能回退到改之前的版本 ✓
Git：能告诉你改完后对不对 ✗
↓
测试：能告诉你改完后对不对 ✓
```

**一句话总结**：Git 是后悔药，测试是体检报告。你需要的是在"吃药"之前就知道有没有病。

### 1.2 没有测试的后果

```python
# 假设你改了 Trainer.train_classifier 的代码
# 没有测试时：
# 1. 你手动跑一下 → "好像没报错"
# 2. 上线后发现 SVM 训练结果全是 0
# 3. 回滚代码，排查半天才发现是数据标准化写错了

# 有测试时：
# 1. pytest 一键运行 → 发现 test_train_svm 失败
# 2. 立刻定位问题，5 分钟修复
```

### 1.3 测试的核心价值

| 价值 | 说明 |
|------|------|
| **防回归** | 改 A 不会悄悄破坏 B |
| **敢于重构** | 没测试改代码是赌博，有测试是工程 |
| **文档作用** | 测试用例 = 最准确的代码使用说明 |
| **加速迭代** | 改完跑测试，秒级反馈，不用手动验证 |
| **降低恐惧** | 不怕改代码，不怕加功能 |

## 二、pytest 基础

### 2.1 安装

```bash
# 开发依赖已包含 pytest
pip install pytest

# 或从 requirements 安装
pip install -r requirements.txt
```

### 2.2 第一个测试

创建 `tests/test_hello.py`：

```python
"""最简单的测试示例"""

def test_addition():
    """测试加法"""
    result = 1 + 1
    assert result == 2

def test_string():
    """测试字符串"""
    result = "hello" + " " + "world"
    assert result == "hello world"
```

运行：

```bash
pytest tests/test_hello.py -v
```

输出：

```
tests/test_hello.py::test_addition PASSED ✓
tests/test_hello.py::test_string PASSED ✓
```

### 2.3 测试的基本结构

每个测试遵循 **AAA 模式**：

```python
def test_xxx():
    # 1. 准备 (Arrange)
    input_data = ...

    # 2. 执行 (Act)
    result = function_under_test(input_data)

    # 3. 断言 (Assert)
    assert result == expected
```

## 三、pytest 核心功能

### 3.1 断言 (assert)

```python
# 相等
assert result == expected

# 不相等
assert result != unexpected

# 真/假
assert condition is True
assert condition is False

# 包含
assert "error" in error_message

# 空/非空
assert result is not None
assert len(result) > 0

# 数值范围
assert 0 <= accuracy <= 1

# 异常
with pytest.raises(ValueError):
    function_that_should_fail()
```

### 3.2 测试夹具 (Fixture)

Fixture 用于准备测试数据，避免重复代码：

```python
import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """准备测试数据"""
    X = np.random.randn(100, 71)
    y = np.random.randint(0, 5, 100)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    """准备训练好的模型"""
    X, y = sample_data
    model_data, acc = Trainer.train_classifier(X, y, 'svm')
    return model_data

# 使用 fixture
def test_model_accuracy(trained_model):
    """测试模型准确率"""
    assert trained_model is not None

def test_prediction(trained_model, sample_data):
    """测试预测功能"""
    X, y = sample_data
    scaler = trained_model['scaler']
    X_scaled = scaler.transform(X[:10])
    predictions = trained_model['model'].predict(X_scaled)
    assert len(predictions) == 10
```

### 3.3 参数化测试

同一测试用不同数据运行多次：

```python
@pytest.mark.parametrize("model_type", ['svm', 'rf', 'mlp'])
def test_classifier_training(model_type):
    """测试所有分类器模型"""
    X = np.random.randn(100, 71)
    y = np.random.randint(0, 5, 100)

    model_data, accuracy = Trainer.train_classifier(X, y, model_type)

    assert model_data is not None
    assert 0 <= accuracy <= 1
```

运行结果：

```
tests/test_trainer.py::test_classifier_training[svm] PASSED ✓
tests/test_trainer.py::test_classifier_training[rf] PASSED ✓
tests/test_trainer.py::test_classifier_training[mlp] PASSED ✓
```

### 3.4 测试类

相关测试组织在一起：

```python
class TestTrainer:
    """测试 Trainer 模块"""

    @pytest.fixture
    def sample_data(self):
        X = np.random.randn(100, 71)
        y = np.random.randint(0, 5, 100)
        return X, y

    def test_train_svm(self, sample_data):
        """测试 SVM 训练"""
        X, y = sample_data
        model_data, acc = Trainer.train_classifier(X, y, 'svm')
        assert acc > 0

    def test_train_rf(self, sample_data):
        """测试随机森林训练"""
        X, y = sample_data
        model_data, acc = Trainer.train_classifier(X, y, 'rf')
        assert acc > 0

    def test_unknown_model_raises(self, sample_data):
        """测试未知模型报错"""
        X, y = sample_data
        with pytest.raises(ValueError):
            Trainer.train_classifier(X, y, 'unknown')
```

## 四、为本项目编写测试

### 4.1 测试策略

按模块优先级编写测试：

| 优先级 | 模块 | 原因 |
|--------|------|------|
| P0 | `trainer.py` | 核心训练逻辑，影响所有模型 |
| P0 | `data_loader.py` | 数据加载，影响所有流程 |
| P1 | `feature_extractor.py` | 特征提取，已有部分测试 |
| P1 | `metrics.py` | 评估指标，逻辑简单 |
| P2 | `hand_detector.py` | 依赖 MediaPipe，需 mock |
| P2 | `models/` | 深度学习模型，需 GPU 测试 |

### 4.2 测试示例：Trainer

创建 `tests/test_trainer.py`：

```python
"""测试 src/training/trainer.py — 模型训练器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.training.trainer import Trainer


class TestTrainerClassifier:
    """测试传统机器学习分类器训练"""

    @pytest.fixture
    def sample_data(self):
        """准备分类器测试数据"""
        np.random.seed(42)
        X = np.random.randn(100, 71)  # 100 个样本，71 维特征
        y = np.random.randint(0, 5, 100)  # 5 个类别
        return X, y

    def test_train_svm_returns_model(self, sample_data):
        """正常路径：训练 SVM 应返回模型和准确率"""
        X, y = sample_data

        model_data, accuracy = Trainer.train_classifier(X, y, 'svm')

        assert model_data is not None
        assert 'model' in model_data
        assert 'scaler' in model_data
        assert model_data['type'] == 'svm'
        assert 0 <= accuracy <= 1

    def test_train_rf_returns_model(self, sample_data):
        """正常路径：训练随机森林应返回模型和准确率"""
        X, y = sample_data

        model_data, accuracy = Trainer.train_classifier(X, y, 'rf')

        assert model_data is not None
        assert model_data['type'] == 'rf'
        assert 0 <= accuracy <= 1

    def test_train_mlp_returns_model(self, sample_data):
        """正常路径：训练 MLP 应返回模型和准确率"""
        X, y = sample_data

        model_data, accuracy = Trainer.train_classifier(X, y, 'mlp')

        assert model_data is not None
        assert model_data['type'] == 'mlp'
        assert 0 <= accuracy <= 1

    def test_train_unknown_model_raises(self, sample_data):
        """异常路径：未知模型应报错"""
        X, y = sample_data

        with pytest.raises(ValueError, match="未知模型类型"):
            Trainer.train_classifier(X, y, 'unknown_model')

    def test_scaler_is_fitted(self, sample_data):
        """验证 scaler 已拟合训练数据"""
        X, y = sample_data

        model_data, _ = Trainer.train_classifier(X, y, 'svm')
        scaler = model_data['scaler']

        # scaler 应有 mean_ 和 scale_ 属性
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')

    def test_prediction_works(self, sample_data):
        """验证预测功能可用"""
        X, y = sample_data

        model_data, _ = Trainer.train_classifier(X, y, 'svm')
        model = model_data['model']
        scaler = model_data['scaler']

        X_test = scaler.transform(X[:5])
        predictions = model.predict(X_test)

        assert len(predictions) == 5
        assert all(0 <= p < 5 for p in predictions)


class TestTrainerDeepLearning:
    """测试深度学习模型训练"""

    @pytest.fixture
    def sequence_data(self):
        """准备序列测试数据"""
        np.random.seed(42)
        X = np.random.randn(50, 30, 71)  # 50 个样本，30 帧，71 维特征
        y = np.random.randint(0, 3, 50)  # 3 个类别
        return X, y

    def test_train_lstm(self, sequence_data):
        """正常路径：训练 LSTM 应成功"""
        from src.models.lstm_model import LSTMModel

        X, y = sequence_data

        model, accuracy = Trainer.train_deep_learning(
            X, y, LSTMModel,
            epochs=2,  # 测试用少量轮次
            batch_size=16
        )

        assert model is not None
        assert 0 <= accuracy <= 1

    def test_train_transformer(self, sequence_data):
        """正常路径：训练 Transformer 应成功"""
        from src.models.transformer_model import TransformerModel

        X, y = sequence_data

        model, accuracy = Trainer.train_deep_learning(
            X, y, TransformerModel,
            epochs=2,
            batch_size=16
        )

        assert model is not None
        assert 0 <= accuracy <= 1

    def test_evaluate_model(self, sequence_data):
        """测试模型评估功能"""
        from src.models.lstm_model import LSTMModel

        X, y = sequence_data
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]

        model, _ = Trainer.train_deep_learning(
            X_train, y_train, LSTMModel,
            epochs=2, batch_size=16
        )

        accuracy = Trainer.evaluate_model(model, X_test, y_test)

        assert 0 <= accuracy <= 1
```

### 4.3 测试示例：DataLoader

创建 `tests/test_data_loader.py`：

```python
"""测试 src/utils/data_loader.py — 数据加载器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tempfile
from src.utils.data_loader import DataLoader, parse_person_id


class TestParsePersonId:
    """测试人员 ID 解析"""

    def test_standard_format(self):
        """标准格式：F_001.npy → F"""
        assert parse_person_id('F_001.npy') == 'F'

    def test_long_id(self):
        """长 ID：user001_005.npy → user001"""
        assert parse_person_id('user001_005.npy') == 'user001'

    def test_meta_file_returns_none(self):
        """元信息文件应返回 None"""
        assert parse_person_id('F_001_meta.npy') is None

    def test_no_underscore(self):
        """无下划线格式应返回 unknown"""
        assert parse_person_id('unknown.npy') == 'unknown'


class TestDataLoader:
    """测试数据加载器"""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """创建临时测试数据目录"""
        # 创建词汇目录
        for word in ['hello', 'world']:
            word_dir = tmp_path / word
            word_dir.mkdir()

            # 创建测试数据文件
            for i in range(3):
                X = np.random.randn(30, 71).astype(np.float32)
                np.save(word_dir / f'F_{i+1:03d}.npy', X)

        return tmp_path

    def test_load_data(self, temp_data_dir):
        """测试加载特征数据"""
        loader = DataLoader()

        X, y, class_labels = loader.load_data(str(temp_data_dir))

        assert X.shape == (6, 30, 71)  # 6 个样本
        assert len(y) == 6
        assert len(class_labels) == 2
        assert 'hello' in class_labels
        assert 'world' in class_labels

    def test_load_data_with_persons(self, temp_data_dir):
        """测试加载数据并返回人员 ID"""
        loader = DataLoader()

        X, y, class_labels, person_ids = loader.load_data(
            str(temp_data_dir), return_persons=True
        )

        assert len(person_ids) == 6
        assert all(pid == 'F' for pid in person_ids)

    def test_load_sequence_data(self, temp_data_dir):
        """测试加载序列数据"""
        loader = DataLoader()

        X, y, class_labels = loader.load_sequence_data(
            str(temp_data_dir), max_length=30
        )

        assert X.shape[0] == 6  # 6 个样本
        assert X.shape[1] == 30  # 30 帧
        assert X.shape[2] == 71  # 71 维特征


class TestSplitByPerson:
    """测试按人员划分数据集"""

    def test_basic_split(self):
        """测试基本划分功能"""
        loader = DataLoader()

        X = np.random.randn(10, 71)
        y = np.random.randint(0, 3, 10)
        person_ids = np.array(['A'] * 4 + ['B'] * 3 + ['C'] * 3)

        result = loader.split_by_person(
            X, y, person_ids,
            train_persons=['A', 'B'],
            val_persons=['C']
        )

        assert result['train'] is not None
        assert result['val'] is not None
        assert result['test'] is None

        X_train, y_train = result['train']
        assert X_train.shape[0] == 7  # A + B 的数据

    def test_auto_split_persons(self):
        """测试自动按人员划分"""
        loader = DataLoader()

        person_ids = np.array(['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'])

        splits = loader.auto_split_persons(person_ids)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        assert len(splits['train']) > 0
```

### 4.4 测试示例：Metrics

创建 `tests/test_metrics.py`：

```python
"""测试 src/utils/metrics.py — 评估指标"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.utils.metrics import Metrics


class TestMetrics:
    """测试评估指标计算"""

    @pytest.fixture
    def metrics(self):
        return Metrics()

    @pytest.fixture
    def sample_labels(self):
        """准备测试标签"""
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 0, 1, 2, 2, 1, 0, 1, 2, 0])
        return y_true, y_pred

    def test_accuracy(self, metrics, sample_labels):
        """测试准确率计算"""
        y_true, y_pred = sample_labels

        result = metrics.calculate_metrics(y_true, y_pred)

        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
        # 正确预测：0,0,1,2,2,1,2,0 = 7/10 = 0.7
        assert abs(result['accuracy'] - 0.7) < 0.01

    def test_precision(self, metrics, sample_labels):
        """测试精确率计算"""
        y_true, y_pred = sample_labels

        result = metrics.calculate_metrics(y_true, y_pred)

        assert 'precision' in result
        assert 0 <= result['precision'] <= 1

    def test_recall(self, metrics, sample_labels):
        """测试召回率计算"""
        y_true, y_pred = sample_labels

        result = metrics.calculate_metrics(y_true, y_pred)

        assert 'recall' in result
        assert 0 <= result['recall'] <= 1

    def test_f1(self, metrics, sample_labels):
        """测试 F1 分数计算"""
        y_true, y_pred = sample_labels

        result = metrics.calculate_metrics(y_true, y_pred)

        assert 'f1' in result
        assert 0 <= result['f1'] <= 1

    def test_perfect_prediction(self, metrics):
        """完美预测的指标"""
        y = np.array([0, 1, 2, 0, 1])

        result = metrics.calculate_metrics(y, y)

        assert result['accuracy'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1'] == 1.0

    def test_random_prediction(self, metrics):
        """随机预测的指标应在合理范围"""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)

        result = metrics.calculate_metrics(y_true, y_pred)

        # 随机预测准确率应在 0.2-0.5 之间
        assert 0.2 <= result['accuracy'] <= 0.5
```

## 五、运行测试

### 5.1 基本命令

```bash
# 运行所有测试
pytest

# 运行指定文件
pytest tests/test_trainer.py

# 运行指定类
pytest tests/test_trainer.py::TestTrainerClassifier

# 运行指定函数
pytest tests/test_trainer.py::TestTrainerClassifier::test_train_svm

# 显示详细输出
pytest -v

# 显示 print 输出
pytest -s

# 只运行上次失败的测试
pytest --lf

# 测试失败时停止
pytest -x
```

### 5.2 生成测试报告

```bash
# 生成 HTML 报告
pytest --html=report.html

# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 查看覆盖率
pytest --cov=src
```

### 5.3 测试结果解读

```
tests/test_trainer.py::TestTrainerClassifier::test_train_svm PASSED ✓
tests/test_trainer.py::TestTrainerClassifier::test_train_rf PASSED ✓
tests/test_trainer.py::TestTrainerClassifier::test_unknown_model FAILED ✗

======================== 2 passed, 1 failed ========================
```

- `PASSED` ✓：测试通过
- `FAILED` ✗：测试失败，查看具体错误
- `ERROR`：测试本身有问题

## 六、最佳实践

### 6.1 测试命名

```python
# 好的命名：清楚表达测试意图
def test_train_svm_returns_model_and_accuracy()
def test_empty_data_raises_value_error()
def test_scaler_is_fitted_on_training_data()

# 不好的命名：看不出在测什么
def test_svm()
def test_data()
def test_1()
```

### 6.2 测试结构

```python
class TestXxx:
    """测试某个模块"""

    @pytest.fixture
    def setup(self):
        """准备工作"""
        pass

    def test_normal_case(self):
        """正常路径"""
        pass

    def test_edge_case_empty(self):
        """边界值：空输入"""
        pass

    def test_edge_case_max(self):
        """边界值：最大值"""
        pass

    def test_error_case(self):
        """异常路径"""
        pass
```

### 6.3 避免的反模式

```python
# ❌ 不要在测试中包含业务逻辑
def test_something():
    result = complex_calculation()
    if result > 0:
        assert True  # 这不是测试，是隐藏错误

# ✅ 断言应该是明确的
def test_something():
    result = complex_calculation()
    assert result == expected_value

# ❌ 不要测试实现细节
def test_internal_method():
    obj._private_method()  # 私有方法可能随时改变

# ✅ 测试公开接口
def test_public_behavior():
    result = obj.public_method()  # 测试行为，不测实现
```

## 七、常见问题

### Q1: 测试运行很慢怎么办？

```bash
# 只运行快速测试
pytest -m "not slow"

# 并行运行（需安装 pytest-xdist）
pytest -n auto
```

### Q2: 测试之间有依赖怎么办？

```python
# ❌ 不要让测试之间有依赖
def test_a():
    global_data = prepare_data()

def test_b():
    use(global_data)  # 依赖 test_a

# ✅ 每个测试独立
@pytest.fixture
def data():
    return prepare_data()

def test_a(data):
    use(data)

def test_b(data):
    use(data)
```

### Q3: 如何测试需要网络/文件的代码？

```python
# 使用 mock 模拟外部依赖
from unittest.mock import patch, MagicMock

@patch('src.detection.hand_detector.cv2')
def test_detector_with_mock_cv2(mock_cv2):
    mock_cv2.VideoCapture.return_value.read.return_value = (True, np.zeros((480, 640, 3)))
    # 测试逻辑...
```

---

*最后更新：2026-06-11*
