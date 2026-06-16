"""
第7阶段：特征工程入门

本脚本帮助您学习如何从关键点数据中提取有意义的特征。
"""

import numpy as np

def section_1_relative_coordinates():
    """7.1 相对坐标计算"""
    print("\n" + "=" * 50)
    print("7.1 相对坐标计算")
    print("=" * 50)

    print("""
为什么需要相对坐标？
-------------------
原始关键点坐标是绝对像素坐标，会受到：
- 手在画面中的位置影响
- 摄像头距离影响
- 画面大小影响

相对坐标可以消除这些影响，使特征更具鲁棒性。
""")

    # 模拟手部关键点数据 (21个关键点)
    hand_landmarks = np.array([
        [100, 200, 0],   # 手腕 (索引0)
        [120, 180, -5],  # 拇指根部 (索引1)
        [140, 160, -10], # 拇指第一关节 (索引2)
        [155, 145, -15], # 拇指第二关节 (索引3)
        [165, 130, -20], # 拇指指尖 (索引4)
        [110, 170, -3],  # 食指根部 (索引5)
        [105, 140, -8],  # 食指第一关节 (索引6)
        [100, 110, -13], # 食指中间关节 (索引7)
        [95, 85, -18],   # 食指指尖 (索引8)
        # ... 省略其他关键点
    ])

    print(f"原始关键点形状: {hand_landmarks.shape}")
    print(f"手腕坐标: {hand_landmarks[0]}")

    # 计算相对坐标（相对于手腕）
    wrist = hand_landmarks[0]
    relative_coords = hand_landmarks - wrist
    print(f"\n相对坐标（相对于手腕）:")
    print(f"手腕相对坐标: {relative_coords[0]}")
    print(f"拇指指尖相对坐标: {relative_coords[4]}")
    print(f"食指指尖相对坐标: {relative_coords[8]}")

def section_2_finger_lengths():
    """7.2 手指长度计算"""
    print("\n" + "=" * 50)
    print("7.2 手指长度计算")
    print("=" * 50)

    # 手指关键点索引
    finger_indices = {
        'thumb': [0, 1, 2, 3, 4],    # 拇指
        'index': [0, 5, 6, 7, 8],     # 食指
        'middle': [0, 9, 10, 11, 12], # 中指
        'ring': [0, 13, 14, 15, 16],  # 无名指
        'pinky': [0, 17, 18, 19, 20], # 小指
    }

    # 模拟手部关键点
    hand_landmarks = np.random.rand(21, 3) * 100

    print("手指长度计算:")
    for finger, indices in finger_indices.items():
        # 计算手指长度（根部到指尖的欧氏距离）
        root = hand_landmarks[indices[1]]  # 手指根部
        tip = hand_landmarks[indices[-1]]  # 手指指尖
        length = np.linalg.norm(tip - root)
        print(f"  {finger}: {length:.2f}")

    # 计算手掌大小
    palm_width = np.linalg.norm(hand_landmarks[17] - hand_landmarks[5])
    print(f"\n手掌宽度: {palm_width:.2f}")

    # 计算归一化手指长度
    print("\n归一化手指长度（相对于手掌宽度）:")
    for finger, indices in finger_indices.items():
        root = hand_landmarks[indices[1]]
        tip = hand_landmarks[indices[-1]]
        length = np.linalg.norm(tip - root) / palm_width
        print(f"  {finger}: {length:.2f}")

def section_3_joint_angles():
    """7.3 关节角度计算"""
    print("\n" + "=" * 50)
    print("7.3 关节角度计算")
    print("=" * 50)

    print("""
关节角度是手部姿态的重要特征。
我们可以通过计算向量之间的夹角来得到关节角度。
""")

    def calculate_angle(a, b, c):
        """
        计算关节角度（点b处的角度）

        参数:
            a: 第一个点坐标
            b: 关节点坐标
            c: 第三个点坐标

        返回:
            角度（弧度）
        """
        # 计算向量
        vec1 = a - b
        vec2 = c - b

        # 计算夹角
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # 防止除零错误
        if norm1 == 0 or norm2 == 0:
            return 0

        cos_angle = dot_product / (norm1 * norm2)
        # 限制在[-1, 1]范围内
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

        return angle

    # 模拟手指关键点
    # 食指: 根部(5) -> 第一关节(6) -> 中间关节(7) -> 指尖(8)
    finger_points = np.array([
        [100, 200, 0],   # 根部 (点5)
        [100, 150, -5],  # 第一关节 (点6)
        [100, 100, -10], # 中间关节 (点7)
        [100, 50, -15],  # 指尖 (点8)
    ])

    # 计算关节角度
    angle1 = calculate_angle(finger_points[0], finger_points[1], finger_points[2])
    angle2 = calculate_angle(finger_points[1], finger_points[2], finger_points[3])

    print(f"食指关节角度:")
    print(f"  第一关节角度: {np.degrees(angle1):.1f} 度")
    print(f"  中间关节角度: {np.degrees(angle2):.1f} 度")

    # 弯曲手指时的角度变化
    print("\n模拟手指弯曲:")
    finger_bent = np.array([
        [100, 200, 0],   # 根部
        [100, 150, -5],  # 第一关节
        [90, 110, -10],  # 中间关节（弯曲）
        [80, 80, -15],   # 指尖（弯曲）
    ])

    angle1_bent = calculate_angle(finger_bent[0], finger_bent[1], finger_bent[2])
    angle2_bent = calculate_angle(finger_bent[1], finger_bent[2], finger_bent[3])

    print(f"  弯曲后第一关节角度: {np.degrees(angle1_bent):.1f} 度")
    print(f"  弯曲后中间关节角度: {np.degrees(angle2_bent):.1f} 度")

def section_4_feature_vector_construction():
    """7.4 构建特征向量"""
    print("\n" + "=" * 50)
    print("7.4 构建特征向量")
    print("=" * 50)

    print("""
项目中有两种特征向量，用途不同：

1. 原始关键点向量（171维）— 用于深度学习模型（LSTM、Transformer）
   - 左手关键点: 21点 × 3坐标 = 63维
   - 右手关键点: 21点 × 3坐标 = 63维
   - 姿态关键点: 15点 × 3坐标 = 45维
   - 总计: 171维

2. 提取后特征向量（71维）— 用于传统ML模型（SVM、RF、MLP）
   - 相对坐标: 21点 × 3坐标 = 63维（相对于手腕）
   - 手指长度: 4维（食指、中指、无名指、小指）
   - 关节角度: 4维（4个手指根部角度）
   - 总计: 71维

为什么要区分？
- 深度学习模型可以自动从原始序列中学习特征
- 传统ML模型需要手工设计的特征才能取得好效果
""")

    # 模拟数据
    left_hand = np.random.rand(21, 3)
    right_hand = np.random.rand(21, 3)
    pose = np.random.rand(15, 3)

    # 构建171维原始关键点向量
    raw_vector = np.concatenate([
        left_hand.flatten(),
        right_hand.flatten(),
        pose.flatten()
    ])

    print(f"左手形状: {left_hand.shape} -> 展平后: {left_hand.flatten().shape}")
    print(f"右手形状: {right_hand.shape} -> 展平后: {right_hand.flatten().shape}")
    print(f"姿态形状: {pose.shape} -> 展平后: {pose.flatten().shape}")
    print(f"原始关键点向量维度: {raw_vector.shape[0]} (用于LSTM/Transformer)")

    # 构建71维提取特征向量
    wrist = left_hand[0]
    relative_coords = (left_hand - wrist).flatten()  # 63维

    finger_lengths = []
    fingers = [left_hand[5:9], left_hand[9:13], left_hand[13:17], left_hand[17:21]]
    for finger in fingers:
        length = np.linalg.norm(finger[-1] - finger[0])
        finger_lengths.append(length)

    extracted_features = np.concatenate([
        relative_coords,                # 63维
        np.array(finger_lengths),       # 4维
        np.random.rand(4)               # 4维（关节角度）
    ])

    print(f"\n提取特征向量维度: {extracted_features.shape[0]} (用于SVM/RF/MLP)")
    print(f"  - 相对坐标: 63维")
    print(f"  - 手指长度: 4维")
    print(f"  - 关节角度: 4维")

def section_5_practical_demo():
    """7.5 实战演示"""
    print("\n" + "=" * 50)
    print("7.5 实战演示")
    print("=" * 50)

    # 添加项目路径
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.detection.hand_detector import HolisticDetector
        print("[OK] 成功导入HolisticDetector")
    except ImportError as e:
        print(f"[ERROR] 无法导入HolisticDetector: {e}")
        return

    print("\n特征提取流程演示:")
    print("1. 初始化检测器")
    print("2. 检测图像获取关键点")
    print("3. 使用get_landmarks()获取171维特征向量")
    print("4. 用于模型训练或预测")

    print("\n示例代码:")
    print("""
detector = HolisticDetector()
results = detector.detect(frame)
landmarks = detector.get_landmarks(results, frame.shape)
# landmarks 就是171维特征向量
""")

def main():
    print("=" * 60)
    print("第7阶段：特征工程入门")
    print("=" * 60)

    section_1_relative_coordinates()
    section_2_finger_lengths()
    section_3_joint_angles()
    section_4_feature_vector_construction()
    section_5_practical_demo()

    print("\n" + "=" * 60)
    print("特征工程学习完成！")
    print("下一步：运行 08_svm_intro.py 学习机器学习")
    print("=" * 60)

if __name__ == '__main__':
    main()
