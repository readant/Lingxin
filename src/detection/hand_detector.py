"""
HandDetector - MediaPipe 手部关键点检测模块

本文件实现了基于 MediaPipe 的手部和姿态关键点检测功能。

包含三个检测器类：
1. HandDetector: 手部关键点检测（21个关键点/只手）
2. PoseDetector: 姿态关键点检测（上半身15个关键点）
3. HolisticDetector: 综合检测器，同时检测手部和姿态

关键点数据格式：
- 手部: (21, 3) - 每只手21个关键点，每个点包含(x, y, z)坐标
- 姿态: (15, 3) - 上半身15个关键点
- 综合: (171,) - 左手63 + 右手63 + 姿态45 = 171维

使用示例：
>>> detector = HolisticDetector()
>>> results = detector.detect(frame)
>>> landmarks = detector.get_landmarks(results, frame.shape)
>>> frame = detector.draw_landmarks(frame, results)
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    手部关键点检测器

    使用 MediaPipe Hands 模型检测手部关键点。
    支持同时检测多只手（默认最多2只）。
    """

    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化手部检测器

        Args:
            static_image_mode (bool, optional): 是否为静态图像模式. Defaults to False.
                - False: 视频流模式，使用跟踪
                - True: 静态图像模式，每次都重新检测
            max_num_hands (int, optional): 最大检测手数. Defaults to 2.
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
            min_tracking_confidence (float, optional): 跟踪置信度阈值. Defaults to 0.5.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, image):
        """
        检测手部关键点

        Args:
            image: 输入图像（BGR格式）

        Returns:
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList: 检测结果
        """
        # MediaPipe 要求输入RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    def get_landmarks(self, results, image_shape):
        """
        从检测结果中提取关键点坐标

        Args:
            results: 检测结果对象
            image_shape (tuple): 图像形状 (height, width, channels)

        Returns:
            np.ndarray: 关键点数组，形状为(n_hands, 21, 3)
        """
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmark = []
                for lm in hand_landmarks.landmark:
                    # 将归一化坐标转换为像素坐标
                    x = lm.x * image_shape[1]  # 宽度方向
                    y = lm.y * image_shape[0]  # 高度方向
                    z = lm.z                    # 深度（相对于手腕）
                    hand_landmark.append([x, y, z])
                landmarks.append(hand_landmark)
        return np.array(landmarks)

    def draw_landmarks(self, image, results):
        """
        在图像上绘制手部关键点和骨架

        Args:
            image: 输入图像（BGR格式）
            results: 检测结果对象

        Returns:
            绘制了关键点的图像
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return image

    def close(self):
        """关闭检测器，释放资源"""
        self.hands.close()


class PoseDetector:
    """
    姿态关键点检测器

    使用 MediaPipe Pose 模型检测人体姿态关键点。
    仅提取上半身15个关键点（索引0~14）。
    """

    def __init__(self, static_image_mode=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化姿态检测器

        Args:
            static_image_mode (bool, optional): 是否为静态图像模式. Defaults to False.
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
            min_tracking_confidence (float, optional): 跟踪置信度阈值. Defaults to 0.5.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, image):
        """
        检测姿态关键点

        Args:
            image: 输入图像（BGR格式）

        Returns:
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList: 检测结果
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def get_landmarks(self, results, image_shape):
        """
        从检测结果中提取上半身关键点坐标

        Args:
            results: 检测结果对象
            image_shape (tuple): 图像形状

        Returns:
            np.ndarray: 关键点数组，形状为(15, 3)
        """
        landmarks = []
        if results.pose_landmarks:
            # 只提取上半身15个关键点（索引0~14）
            for i in range(15):
                lm = results.pose_landmarks.landmark[i]
                x = lm.x * image_shape[1]
                y = lm.y * image_shape[0]
                z = lm.z
                landmarks.append([x, y, z])
        return np.array(landmarks)

    def draw_landmarks(self, image, results):
        """
        在图像上绘制姿态关键点和骨架

        Args:
            image: 输入图像（BGR格式）
            results: 检测结果对象

        Returns:
            绘制了关键点的图像
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
        return image

    def close(self):
        """关闭检测器，释放资源"""
        self.pose.close()


class HolisticDetector:
    """
    综合检测器

    同时检测手部和姿态关键点，将两者的数据合并为一个特征向量。
    用于手语数据采集场景。
    """

    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化综合检测器

        Args:
            static_image_mode (bool, optional): 是否为静态图像模式. Defaults to False.
            max_num_hands (int, optional): 最大检测手数. Defaults to 2.
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
            min_tracking_confidence (float, optional): 跟踪置信度阈值. Defaults to 0.5.
        """
        self.hand_detector = HandDetector(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.pose_detector = PoseDetector(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, image):
        """
        同时检测手部和姿态关键点

        Args:
            image: 输入图像（BGR格式）

        Returns:
            tuple: (hand_results, pose_results)
        """
        hand_results = self.hand_detector.detect(image)
        pose_results = self.pose_detector.detect(image)
        return hand_results, pose_results

    def get_landmarks(self, results, image_shape):
        """
        提取综合关键点向量（171维）

        输出格式：
        - 前63维: 左手关键点（21个点 × 3坐标）
        - 接下来63维: 右手关键点（21个点 × 3坐标）
        - 最后45维: 姿态关键点（15个点 × 3坐标）

        Args:
            results (tuple): (hand_results, pose_results)
            image_shape (tuple): 图像形状

        Returns:
            np.ndarray: 171维综合特征向量
        """
        hand_results, pose_results = results

        # 处理手部关键点
        hand_landmarks = self.hand_detector.get_landmarks(hand_results, image_shape)
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if len(hand_landmarks) > 0:
            # 假设第一个是左手，第二个是右手
            if len(hand_landmarks) >= 1:
                left_hand = hand_landmarks[0]
            if len(hand_landmarks) >= 2:
                right_hand = hand_landmarks[1]

        # 处理姿态关键点
        pose_landmarks = self.pose_detector.get_landmarks(pose_results, image_shape)
        if len(pose_landmarks) == 0:
            pose_landmarks = np.zeros((15, 3))

        # 组合为171维向量
        combined = np.concatenate([
            left_hand.flatten(),      # 63维
            right_hand.flatten(),     # 63维
            pose_landmarks.flatten()  # 45维
        ])
        return combined

    def draw_landmarks(self, image, results):
        """
        在图像上同时绘制手部和姿态关键点

        Args:
            image: 输入图像（BGR格式）
            results (tuple): (hand_results, pose_results)

        Returns:
            绘制了关键点的图像
        """
        hand_results, pose_results = results
        image = self.hand_detector.draw_landmarks(image, hand_results)
        image = self.pose_detector.draw_landmarks(image, pose_results)
        return image

    def close(self):
        """关闭所有检测器，释放资源"""
        self.hand_detector.close()
        self.pose_detector.close()