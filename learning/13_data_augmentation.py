"""
第13阶段：数据增强 - augmentation.py

本脚本帮助您了解项目中的关键点序列数据增强方法。
"""

import numpy as np
import os

def section_1_why_augmentation():
    """13.1 为什么需要数据增强"""
    print("\n" + "=" * 50)
    print("13.1 为什么需要数据增强")
    print("=" * 50)

    print("""
数据增强的目的：
---------------
- 增加训练数据的多样性
- 提高模型的泛化能力
- 缓解过拟合问题
- 模拟真实场景的变化

手语数据的特点：
---------------
- 同一个手势，不同人做出来有差异
- 同一个人，每次做的速度和幅度也不同
- 摄像头距离、角度会影响关键点坐标

数据增强模拟这些变化，让模型学会"认手势而不是认人"。
""")

def section_2_augmenter_intro():
    """13.2 KeypointAugmenter 介绍"""
    print("\n" + "=" * 50)
    print("13.2 KeypointAugmenter 介绍")
    print("=" * 50)

    print("""
项目实现了 KeypointAugmenter 类（src/features/augmentation.py）

支持的增强操作：
--------------
1. 随机平移 (translate)
   - 对关键点坐标添加微小偏移
   - 模拟手在画面中位置的变化

2. 随机缩放 (scale)
   - 对关键点进行缩放
   - 模拟摄像头距离的变化

3. 高斯噪声 (noise)
   - 对坐标添加随机噪声
   - 模拟MediaPipe检测误差

4. 随机遮挡 (dropout)
   - 随机丢弃部分关键点
   - 模拟手部被遮挡的情况

5. 时间扭曲 (time_warp)
   - 对序列帧进行插值扭曲
   - 模拟动作速度的变化

参数说明：
---------
  p:                每个增强操作的应用概率（默认0.5）
  translate_range:  平移范围（默认0.02）
  scale_range:      缩放范围（默认0.9-1.1）
  noise_std:        噪声标准差（默认0.01）
  dropout_prob:     遮挡概率（默认0.05）
  time_warp_sigma:  时间扭曲平滑度（默认3.0）
""")

def section_3_practical_demo():
    """13.3 实际演示"""
    print("\n" + "=" * 50)
    print("13.3 实际演示")
    print("=" * 50)

    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.features.augmentation import KeypointAugmenter
        from src.constants import EXTRACTED_FEATURE_DIMS, DEFAULT_SEQUENCE_LENGTH
        print("[OK] 成功导入 KeypointAugmenter")
    except ImportError as e:
        print(f"[ERROR] 无法导入: {e}")
        return

    # 创建模拟序列数据 (30帧, 71维特征)
    np.random.seed(42)
    original = np.random.randn(DEFAULT_SEQUENCE_LENGTH, EXTRACTED_FEATURE_DIMS).astype(np.float32)

    print(f"\n原始序列形状: {original.shape}")
    print(f"原始序列均值: {original.mean():.4f}")
    print(f"原始序列标准差: {original.std():.4f}")

    # 创建增强器（100%应用所有增强）
    augmenter = KeypointAugmenter(p=1.0)

    # 应用增强
    augmented = augmenter(original)

    print(f"\n增强后序列形状: {augmented.shape}")
    print(f"增强后序列均值: {augmented.mean():.4f}")
    print(f"增强后序列标准差: {augmented.std():.4f}")

    # 对比差异
    diff = np.abs(augmented - original)
    print(f"\n增强前后差异:")
    print(f"  最大差异: {diff.max():.4f}")
    print(f"  平均差异: {diff.mean():.4f}")

    # 逐操作演示
    print("\n逐操作演示:")
    print("-" * 40)

    augmenter_single = KeypointAugmenter(p=0.0)

    # 平移
    result = augmenter_single._translate(original.copy())
    print(f"平移: 差异={np.abs(result - original).mean():.4f}")

    # 噪声
    result = augmenter_single._add_noise(original.copy())
    print(f"噪声: 差异={np.abs(result - original).mean():.4f}")

    # 时间扭曲
    result = augmenter_single._time_warp(original.copy())
    print(f"时间扭曲: 差异={np.abs(result - original).mean():.4f}")

    # p=0 时不应有变化
    augmenter_none = KeypointAugmenter(p=0.0)
    result_none = augmenter_none(original.copy())
    print(f"\np=0时无变化: {np.allclose(result_none, original)}")

def section_4_how_to_use():
    """13.4 如何在训练中使用"""
    print("\n" + "=" * 50)
    print("13.4 如何在训练中使用")
    print("=" * 50)

    print("""
在训练中使用数据增强：

  from src.features.augmentation import KeypointAugmenter
  from src.utils.data_loader import SignLanguageDataset, create_dataloaders

  # 创建增强器
  augmenter = KeypointAugmenter(p=0.5)

  # 方法1: 使用便捷函数
  train_loader, val_loader = create_dataloaders(
      X_train, y_train,
      batch_size=32,
      augmenter=augmenter
  )

  # 方法2: 手动创建 Dataset
  dataset = SignLanguageDataset(X_train, y_train, augmenter=augmenter)
  loader = DataLoader(dataset, batch_size=32, shuffle=True)

注意：
- 增强只在训练时应用，验证/测试时不应用
- p=0.5 表示每个增强操作有50%概率被应用
- 2D输入(71维)只支持平移/噪声/时间扭曲
- 3D输入(21,3)支持所有增强操作
""")

def section_5_tips():
    """13.5 增强策略建议"""
    print("\n" + "=" * 50)
    print("13.5 增强策略建议")
    print("=" * 50)

    print("""
增强参数调优建议：
-----------------

1. 数据量少时（<100样本/类）：
   - p 设为 0.7-0.8（更激进的增强）
   - noise_std 设为 0.02（更大的噪声）

2. 数据量适中时（100-500样本/类）：
   - p 设为 0.5（默认值）
   - 保持默认参数

3. 数据量充足时（>500样本/类）：
   - p 设为 0.3（轻度增强）
   - 或者不使用增强

常见问题：
---------
Q: 增强后准确率反而下降？
A: 可能增强太强了，降低 p 或减小参数范围

Q: 训练集准确率高但验证集低？
A: 过拟合，增加增强强度或增加数据量

Q: 增强后模型变慢了？
A: 增强在CPU上运行，可以减少增强操作数量
""")

def main():
    print("=" * 60)
    print("第13阶段：数据增强 - augmentation.py")
    print("=" * 60)

    section_1_why_augmentation()
    section_2_augmenter_intro()
    section_3_practical_demo()
    section_4_how_to_use()
    section_5_tips()

    print("\n" + "=" * 60)
    print("数据增强学习完成！")
    print("学习路线全部完成！可以开始项目实战了。")
    print("=" * 60)

if __name__ == '__main__':
    main()
