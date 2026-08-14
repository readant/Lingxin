"""
第14阶段：实时推理闭环

本脚本演示如何把训练好的模型接入实时摄像头做手语识别，
衔接 tools/inference.py 的真实推理流程，作为整个学习路线的收尾。

对应文档：docs/usage/04-training.md
"""

import sys
import os


def section_1_why_inference():
    """14.1 为什么需要推理闭环"""
    print("\n" + "=" * 50)
    print("14.1 为什么需要推理闭环")
    print("=" * 50)
    print("""
到此为止，你已掌握：采集(10) → 配置(11) → 训练(12) → 增强(13)。
但训练出的模型若不接入摄像头，就无法投入实际使用。

"推理（Inference）" = 用训练好的模型对新的输入做预测。

完整工作流：
  采集 → 预处理 → 训练 → 评估 → 实时推理
  collect  preprocess  train  evaluate  inference

推理环节做的事情：
1. 加载已保存的模型与类别标签
2. 从摄像头逐帧提取 171 维关键点
3. 交给模型预测手势类别
4. 把结果绘制回画面
""")


def section_2_project_inference():
    """14.2 项目真实推理入口"""
    print("\n" + "=" * 50)
    print("14.2 项目真实推理入口")
    print("=" * 50)
    print("""
项目的实时推理由 tools/inference.py 提供，通过命令行即可启动。

常用用法：
  # 传统ML模型（如 SVM，提取 71 维特征后逐帧预测）
  python tools/inference.py --model svm

  # 深度学习模型（LSTM，需要累积 30 帧序列后预测）
  python tools/inference.py --model lstm

  # 指定模型与类别标签路径
  python tools/inference.py --model lstm \\
      --model-path models/lstm_model.pth \\
      --labels data/processed/class_labels.npy

支持的模型类型：svm / rf / mlp / lstm / transformer

运行前需确保：
- 已用 tools/train.py 训练出对应模型（见第12阶段）
- 已生成类别标签 data/processed/class_labels.npy
""")


def section_3_inference_flow():
    """14.3 推理流程详解"""
    print("\n" + "=" * 50)
    print("14.3 推理流程详解")
    print("=" * 50)
    print("""
InferenceRunner（tools/inference.py）的核心流程：

 1. 初始化 HolisticDetector，加载模型与类别标签
    runner = InferenceRunner()

 2. 逐帧处理摄像头画面：
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)               # 镜像（与采集一致）
    results = detector.detect(frame)
    landmarks = detector.get_landmarks(results, frame.shape)

 3. 归一化 x,y 到 [0,1]（与训练时一致）：
    landmarks_norm[0::3] /= w
    landmarks_norm[1::3] /= h

 4. 判断是否有手，再决定预测方式：
    传统ML:   _predict_single  提取特征 → scaler → model.predict
    深度学习: _predict_sequence 累积序列到 max_sequence_length 再预测

 5. 用 PIL 渲染中文预测结果到画面上
""")


def section_4_sequence_vs_single():
    """14.4 单帧 vs 序列预测"""
    print("\n" + "=" * 50)
    print("14.4 单帧 vs 序列预测")
    print("=" * 50)
    print("""
不同模型对输入的粒度要求不同：

传统ML（SVM/RF/MLP）— 单帧预测
  每一帧的 171 维关键点，经 FeatureExtractor 提取后
  直接喂给分类器，立即得到一个预测结果。
  优点：响应快
  缺点：看不到动作的时间演化

深度学习（LSTM/Transformer）— 序列预测
  需要先累积一个动作片段的连续帧（默认 30 帧），
  组成 (30, 171) 的序列，再交给模型判断完整动作。
  优点：能识别动态手势
  缺点：需要缓冲区，响应略慢

这正是为什么第 7 阶段区分 171 维原始向量（深度学习）
与 71 维提取特征（传统ML）的原因。
""")


def section_5_demo_simplified():
    """14.5 简化版推理演示（理解原理）"""
    print("\n" + "=" * 50)
    print("14.5 简化版推理演示（理解原理）")
    print("=" * 50)

    # 仅演示推理的流程骨架，不真正打开摄像头/加载模型
    print("""
以下是一个「不依赖摄像头与模型」的流程演示，
帮助你理解 171 维特征 → 归一化 → 预测 的数据流动：

  import numpy as np

  # 模拟一帧检测到的 171 维关键点（前126维为双手，后45维为姿态）
  landmarks = np.zeros(171)

  # 归一化 x,y 到 [0,1]（index 0,3,6,... 为 x；1,4,7,... 为 y）
  h, w = 480, 640
  landmarks[0::3] /= w
  landmarks[1::3] /= h

  # 判断是否有手：双手 126 维是否接近全零
  has_hand = bool(np.any(np.abs(landmarks[:126]) > 1e-4))
  print(f"是否检测到手: {has_hand}")

  # 传给模型前的形状转换（此处仅示意，不实际加载模型）
  single = landmarks.reshape(1, -1)        # 传统ML: (1, 171)
  sequence = landmarks.reshape(1, 1, 171)  # 深度学习: (1, 1, 171)
  print(f"传统ML输入形状: {single.shape}")
  print(f"深度学习序列形状: {sequence.shape}")
""")

    import numpy as np
    landmarks = np.zeros(171)
    h, w = 480, 640
    landmarks[0::3] /= w
    landmarks[1::3] /= h
    has_hand = bool(np.any(np.abs(landmarks[:126]) > 1e-4))
    print(f"[演示] 是否检测到手: {has_hand}")
    single = landmarks.reshape(1, -1)
    sequence = landmarks.reshape(1, 1, 171)
    print(f"[演示] 传统ML输入形状: {single.shape}")
    print(f"[演示] 深度学习序列形状: {sequence.shape}")


def section_6_launch():
    """14.6 实际运行"""
    print("\n" + "=" * 50)
    print("14.6 实际运行")
    print("=" * 50)
    print("""
要真正看到实时识别效果，请在项目根目录运行：

  conda activate lingxin-gpu
  python tools/inference.py --model svm
  # 或
  python tools/inference.py --model lstm

运行界面：
  - 画面左上角显示识别到的词
  - 按 q / ESC 退出

若提示「无法打开摄像头」，检查：
  1. 摄像头是否被占用
  2. 是否有权限访问
""")


def main():
    print("=" * 60)
    print("第14阶段：实时推理闭环")
    print("=" * 60)

    section_1_why_inference()
    section_2_project_inference()
    section_3_inference_flow()
    section_4_sequence_vs_single()
    section_5_demo_simplified()
    section_6_launch()

    print("\n" + "=" * 60)
    print("实时推理学习完成！")
    print("下一步：运行 15_api_intro.py 了解 Web 部署")
    print("=" * 60)


if __name__ == '__main__':
    main()
