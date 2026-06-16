"""
第12阶段：训练流程 - train.py 和 evaluate.py

本脚本帮助您了解项目中的完整训练和评估流程。
"""

import sys
import os

def section_1_pipeline_overview():
    """12.1 训练流程概览"""
    print("\n" + "=" * 50)
    print("12.1 训练流程概览")
    print("=" * 50)

    print("""
项目训练流程：
-------------

  数据采集 → 预处理 → 训练 → 评估 → 推理
  collect    preprocess  train   evaluate  inference

支持的模型：
-----------
  传统ML:   SVM, 随机森林(RF), MLP
  深度学习: LSTM, Transformer

数据格式：
---------
  传统ML:   提取71维特征 (X.npy)
  深度学习: 171维原始序列 (X_sequence.npy)
""")

def section_2_preprocess():
    """12.2 数据预处理"""
    print("\n" + "=" * 50)
    print("12.2 数据预处理")
    print("=" * 50)

    print("""
预处理工具: tools/preprocess.py

基本用法：
  # 查看数据分布
  python tools/preprocess.py --summary

  # 标准预处理
  python tools/preprocess.py

  # 按人员划分（推荐）
  python tools/preprocess.py --split-by-person \\
      --train-persons L \\
      --test-persons F

预处理内容：
-----------
1. 加载原始 .npy 文件
2. 特征标准化（StandardScaler）
3. 序列对齐（统一为30帧）
4. 按人员划分训练/验证/测试集

输出文件：
---------
  data/processed/
  ├── X.npy / y.npy              # 传统ML用
  ├── X_sequence.npy / y_sequence.npy  # 深度学习用
  ├── X_train.npy / y_train.npy  # 预划分版本
  ├── X_val.npy / y_val.npy
  ├── X_test.npy / y_test.npy
  ├── scaler.pkl                 # 标准化器
  ├── class_labels.npy           # 类别标签
  └── split_info.json            # 划分详情
""")

def section_3_train():
    """12.3 模型训练"""
    print("\n" + "=" * 50)
    print("12.3 模型训练")
    print("=" * 50)

    print("""
训练工具: tools/train.py

基本用法：
  # 训练SVM
  python tools/train.py --model svm

  # 训练LSTM
  python tools/train.py --model lstm

  # 训练Transformer
  python tools/train.py --model transformer

  # 指定参数
  python tools/train.py --model lstm --epochs 100 --lr 0.0005

支持的参数：
-----------
  --model      模型类型 (svm/rf/mlp/lstm/transformer)
  --data       数据目录 (默认: data/processed)
  --save       模型保存目录 (默认: models/)
  --epochs     训练轮数 (默认: 50)
  --batch-size 批次大小 (默认: 32)
  --lr         学习率 (默认: 0.001)
  --patience   早停容忍轮数 (默认: 10)
  --device     训练设备 (cpu/cuda/auto)

训练过程：
---------
1. 加载数据（自动检测预划分格式）
2. 传统ML: StandardScaler + 模型训练
   深度学习: DataLoader + BaseModel.train_model()
3. 保存模型到 models/ 目录
4. 在测试集上评估
""")

def section_4_train_code():
    """12.4 训练代码解析"""
    print("\n" + "=" * 50)
    print("12.4 训练代码解析")
    print("=" * 50)

    print("""
TrainRunner 核心逻辑（tools/train.py）：

  class TrainRunner:
      MODEL_MAP = {
          'svm':         {'type': 'classifier'},
          'rf':          {'type': 'classifier'},
          'mlp':         {'type': 'classifier'},
          'lstm':        {'type': 'deep_learning', 'model_class': LSTMModel},
          'transformer': {'type': 'deep_learning', 'model_class': TransformerModel},
      }

      def run(self, model_type='svm'):
          # 1. 加载数据（自动检测预划分）
          data = self._load_split_data(data_dir, category)

          # 2. 训练
          if category == 'classifier':
              model_data, acc = Trainer.train_classifier(X_train, y_train, model_type)
              Trainer.save_classifier(model_data, model_path)
          else:
              model, acc = Trainer.train_deep_learning(
                  X_train, y_train, model_class,
                  X_val=X_val, y_val=y_val
              )
              model.save(model_path)

          # 3. 测试集评估
          if X_test is not None:
              test_acc = evaluate(model, X_test, y_test)

Trainer 支持的功能（src/training/trainer.py）：
----------------------------------------------
  - train_classifier()     训练传统ML模型
  - train_deep_learning()  训练深度学习模型
  - evaluate_model()       在独立测试集上评估
  - save_classifier()      保存分类器
  - load_classifier()      加载分类器
""")

def section_5_evaluate():
    """12.5 模型评估"""
    print("\n" + "=" * 50)
    print("12.5 模型评估")
    print("=" * 50)

    print("""
评估工具: tools/evaluate.py

使用方法：
  python tools/evaluate.py
  （然后选择模型类型）

评估指标：
---------
  - 准确率 (Accuracy)
  - 精确率 (Precision) — 加权平均
  - 召回率 (Recall) — 加权平均
  - F1分数 (F1-Score) — 加权平均
  - 混淆矩阵 (Confusion Matrix)

评估代码示例：
  from src.utils.metrics import Metrics

  metrics = Metrics()
  results = metrics.calculate_metrics(y_true, y_pred)
  metrics.print_metrics(results)
  metrics.plot_confusion_matrix(y_true, y_pred, class_names)
""")

def section_6_practical():
    """12.6 实际操作指南"""
    print("\n" + "=" * 50)
    print("12.6 实际操作指南")
    print("=" * 50)

    print("""
完整训练流程（以LSTM为例）：

步骤1: 确认数据已采集
  ls data/raw/collected/
  # 应看到各词汇的子目录

步骤2: 预处理
  python tools/preprocess.py --split-by-person --train-persons L

步骤3: 训练
  python tools/train.py --model lstm

步骤4: 查看结果
  # 模型保存在 models/lstm_model.pth
  # 训练日志会输出验证集准确率

步骤5: 推理测试
  python tools/inference.py --model lstm --checkpoint models/lstm_model.pth

提示：
- 首次训练建议用 SVM（快速验证流程）
- 深度学习模型需要更多数据才能发挥优势
- 使用 --split-by-person 防止数据泄露
""")

def main():
    print("=" * 60)
    print("第12阶段：训练流程 - train.py 和 evaluate.py")
    print("=" * 60)

    section_1_pipeline_overview()
    section_2_preprocess()
    section_3_train()
    section_4_train_code()
    section_5_evaluate()
    section_6_practical()

    print("\n" + "=" * 60)
    print("训练流程学习完成！")
    print("下一步：运行 13_data_augmentation.py 学习数据增强")
    print("=" * 60)

if __name__ == '__main__':
    main()
