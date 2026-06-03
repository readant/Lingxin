"""
FeatureExtractor - 手部关键点特征提取器

本文件实现了从MediaPipe检测的21个手部关键点中提取特征向量的功能。

特征提取流程：
1. 相对化处理：将所有关键点坐标减去手腕坐标，实现位置不变性
2. 计算手指长度：计算4根手指（食指、中指、无名指、小指）的长度
3. 计算关节角度：计算4个手指根部的弯曲角度
4. 特征拼接：将所有特征拼接成71维特征向量

特征向量构成：
- 相对关键点坐标：21点 × 3维 = 63维
- 手指长度：4维（食指、中指、无名指、小指）
- 关节角度：4维（4个手指根部角度）
- 总计：71维
"""

import numpy as np


class FeatureExtractor:
    """
    手部关键点特征提取器

    负责将MediaPipe检测到的手部关键点转换为机器学习模型可用的特征向量。
    """

    def __init__(self):
        """初始化特征提取器"""
        pass

    def extract_features(self, landmarks):
        """
        从手部关键点中提取特征向量

        Args:
            landmarks (list): 手部关键点列表，每个元素是形状为(21, 3)的numpy数组
                             包含(x, y, z)坐标，z表示深度信息

        Returns:
            np.ndarray: 71维特征向量
        """
        features = []

        # 遍历每只手的关键点（支持多手检测）
        for hand_landmarks in landmarks:
            # 跳过空的关键点数据
            if len(hand_landmarks) == 0:
                continue

            # 步骤1：相对化处理 - 以手腕为原点
            # 手腕是第0个关键点，这样处理可以消除手部位置的影响
            wrist = hand_landmarks[0]
            relative_landmarks = hand_landmarks - wrist

            # 步骤2：计算手指长度
            # 手指关键点索引：食指(5-8)、中指(9-12)、无名指(13-16)、小指(17-20)
            fingers = [
                hand_landmarks[5:9],   # 食指
                hand_landmarks[9:13],  # 中指
                hand_landmarks[13:17], # 无名指
                hand_landmarks[17:21]  # 小指
            ]

            finger_lengths = []
            for finger in fingers:
                # 计算指尖到指根的欧氏距离
                length = np.linalg.norm(finger[-1] - finger[0])
                finger_lengths.append(length)

            # 步骤3：计算关节角度（手指根部角度）
            angles = self.calculate_angles(hand_landmarks)

            # 步骤4：拼接所有特征
            hand_features = np.concatenate([
                relative_landmarks.flatten(),  # 63维：相对坐标
                finger_lengths,                # 4维：手指长度
                angles                         # 4维：关节角度
            ])
            features.append(hand_features)

        # 如果没有检测到任何手，返回零向量
        if not features:
            return np.zeros(63 + 4 + 4)

        # 如果检测到多只手，取平均值
        return np.mean(features, axis=0)

    def calculate_angles(self, landmarks):
        """
        计算手指根部的弯曲角度

        使用向量点积公式计算角度：
        cos(theta) = (v1 · v2) / (|v1| × |v2|)
        theta = arccos(cos(theta))

        Args:
            landmarks (np.ndarray): 21个手部关键点，形状为(21, 3)

        Returns:
            np.ndarray: 4个角度值（对应4根手指根部）
        """
        angles = []

        # 定义计算角度的关键点索引组合
        # 格式：[手腕, 手指根部, 手指中间]
        finger_angle_points = [
            [0, 1, 2],   # 拇指根部角度（手腕-拇指根-拇指中节）
            [0, 5, 6],   # 食指根部角度（手腕-食指根-食指中节）
            [0, 9, 10],  # 中指根部角度（手腕-中指根-中指中节）
            [0, 13, 14]  # 无名指根部角度（手腕-无名指根-无名指中节）
        ]

        for finger in finger_angle_points:
            # 获取三个关键点的坐标
            a = landmarks[finger[0]]  # 顶点A（手腕）
            b = landmarks[finger[1]]  # 顶点B（手指根部）
            c = landmarks[finger[2]]  # 顶点C（手指中间）

            # 计算向量 BA 和 BC
            v1 = a - b  # 从B指向A的向量
            v2 = c - b  # 从B指向C的向量

            # 使用点积计算余弦值
            # 添加1e-8防止除零错误
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

            # 使用反余弦计算角度（弧度）
            angle = np.arccos(np.clip(cosine_angle, -1, 1))  # 限制范围防止数值误差
            angles.append(angle)

        return np.array(angles)
