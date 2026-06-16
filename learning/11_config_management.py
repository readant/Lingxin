"""
第11阶段：配置管理 - config.py 和 constants.py

本脚本帮助您了解项目中的统一配置管理和共享常量系统。
"""

import sys
import os

def section_1_config_intro():
    """11.1 配置管理简介"""
    print("\n" + "=" * 50)
    print("11.1 配置管理简介")
    print("=" * 50)

    print("""
为什么需要配置管理？
-------------------
项目中有大量路径、超参数和常量，如果分散在各个文件中：
- 修改一个参数需要改动多个文件
- 容易出现不一致
- 难以维护

配置管理解决的问题：
-------------------
- 单一配置来源（Single Source of Truth）
- 集中管理所有路径和参数
- 消除硬编码
- 方便测试和部署

本项目的配置管理：
----------------
src/config.py    — 统一配置管理（dataclass 单例）
src/constants.py — 共享常量（骨架连接、维度常量）
""")

def section_2_config_usage():
    """11.2 config.py 使用方法"""
    print("\n" + "=" * 50)
    print("11.2 config.py 使用方法")
    print("=" * 50)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.config import config
        print("[OK] 成功导入 config")

        print(f"""
全局配置单例（ProjectConfig）：
----------------------------
项目根目录:   {config.project_root}
数据目录:     {config.data_dir}
原始数据:     {config.raw_data_dir}
处理后数据:   {config.processed_data_dir}
模型目录:     {config.models_dir}
词汇表:       {config.vocab_path}

序列参数：
  最大序列长度:     {config.max_sequence_length}
  最小序列长度:     {config.min_sequence_length}
  特征维度:         {config.extracted_feature_dims}

训练参数：
  默认轮数:         {config.default_epochs}
  默认学习率:       {config.default_learning_rate}
  默认批次大小:     {config.default_batch_size}
  早停容忍轮数:     {config.early_stopping_patience}

模型路径示例：
  SVM:    {config.get_model_path('svm')}
  LSTM:   {config.get_model_path('lstm')}
""")

    except ImportError as e:
        print(f"[ERROR] 无法导入 config: {e}")

def section_3_config_api():
    """11.3 config.py 常用API"""
    print("\n" + "=" * 50)
    print("11.3 config.py 常用API")
    print("=" * 50)

    print("""
常用方法：
---------

1. get_model_path(model_type) — 获取模型保存路径
   config.get_model_path('svm')       → models/svm_model.pkl
   config.get_model_path('lstm')      → models/lstm_model.pth
   config.get_model_path('transformer') → models/transformer_model.pth

2. get_data_files(category) — 获取数据文件名
   config.get_data_files('classifier')    → {'X': 'X.npy', 'y': 'y.npy'}
   config.get_data_files('deep_learning') → {'X': 'X_sequence.npy', 'y': 'y_sequence.npy'}

3. ensure_dirs() — 确保目录存在
   config.ensure_dirs()  # 自动创建 data/、models/ 等目录

模型类型列表：
  分类器:     {classifier_models}
  深度学习:   {deep_learning_models}
  全部:       {all_models}
""".format(
        classifier_models=config.classifier_models if 'config' in dir() else "('svm', 'rf', 'mlp')",
        deep_learning_models=config.deep_learning_models if 'config' in dir() else "('lstm', 'transformer')",
        all_models=config.all_models if 'config' in dir() else "('svm', 'rf', 'mlp', 'lstm', 'transformer')"
    ))

def section_4_constants():
    """11.4 constants.py 常量定义"""
    print("\n" + "=" * 50)
    print("11.4 constants.py 常量定义")
    print("=" * 50)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.constants import (
            HAND_CONNECTIONS,
            POSE_CONNECTIONS_UPPER_BODY,
            RAW_HOLISTIC_FEATURES,
            EXTRACTED_FEATURE_DIMS,
            DEFAULT_SEQUENCE_LENGTH,
        )
        print("[OK] 成功导入 constants")

        print(f"""
手部骨架连接：
  HAND_CONNECTIONS: {len(HAND_CONNECTIONS)} 条连接线
  用于绘制手部21个关键点的骨架

姿态骨架连接：
  POSE_CONNECTIONS_UPPER_BODY: {len(POSE_CONNECTIONS_UPPER_BODY)} 条连接线
  用于绘制上半身15个姿态关键点

维度常量：
  RAW_HOLISTIC_FEATURES:    {RAW_HOLISTIC_FEATURES}  (原始关键点维度)
  EXTRACTED_FEATURE_DIMS:   {EXTRACTED_FEATURE_DIMS}  (提取特征维度)
  DEFAULT_SEQUENCE_LENGTH:  {DEFAULT_SEQUENCE_LENGTH}  (默认序列长度)
""")

    except ImportError as e:
        print(f"[ERROR] 无法导入 constants: {e}")

def section_5_practical():
    """11.5 实际应用示例"""
    print("\n" + "=" * 50)
    print("11.5 实际应用示例")
    print("=" * 50)

    print("""
在项目代码中使用配置：

  from src.config import config

  # 加载数据
  import numpy as np
  X = np.load(config.processed_data_dir / 'X.npy')

  # 训练模型
  model_path = config.get_model_path('lstm')

  # 从 constants 导入维度常量
  from src.constants import EXTRACTED_FEATURE_DIMS
  input_dim = EXTRACTED_FEATURE_DIMS  # 71

使用配置的好处：
- 路径变更只需改 config.py 一处
- 不同环境（开发/测试/生产）可覆盖配置
- 消除代码中的魔法数字
""")

def main():
    print("=" * 60)
    print("第11阶段：配置管理 - config.py 和 constants.py")
    print("=" * 60)

    section_1_config_intro()
    section_2_config_usage()
    section_3_config_api()
    section_4_constants()
    section_5_practical()

    print("\n" + "=" * 60)
    print("配置管理学习完成！")
    print("下一步：运行 12_training_pipeline.py 学习训练流程")
    print("=" * 60)

if __name__ == '__main__':
    main()
