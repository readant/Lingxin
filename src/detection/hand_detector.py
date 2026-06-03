"""
HandDetector - MediaPipe 手部关键点检测模块（新版Task API）

本文件实现了基于 MediaPipe Task API 的手部和姿态关键点检测功能。

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

注意：使用本模块需要安装新版MediaPipe：
pip install mediapipe>=0.10.33

首次使用需要下载模型文件：
python learning/download_models.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os

from src.constants import HAND_CONNECTIONS, POSE_CONNECTIONS_UPPER_BODY


def get_model_path():
    """获取模型文件路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(project_root, 'models')
    return models_dir


class HandDetector:
    """
    手部关键点检测器（新版Task API）

    使用 MediaPipe Hands 模型检测手部关键点。
    支持同时检测多只手（默认最多2只）。
    """

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5):
        """
        初始化手部检测器

        Args:
            max_num_hands (int, optional): 最大检测手数. Defaults to 2.
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
        """
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # 获取模型文件路径
        models_dir = get_model_path()
        model_path = os.path.join(models_dir, 'hand_landmarker.task')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请运行以下命令下载模型:\n"
                f"python learning/download_models.py"
            )

        # 使用本地模型文件
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(self.options)

        # 手部关键点连接关系（从共享常量导入）
        self.hand_connections = HAND_CONNECTIONS

    def detect(self, image):
        """
        检测手部关键点

        Args:
            image: 输入图像（BGR格式）

        Returns:
            HandLandmarkerResult: 检测结果对象
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        return results

    def get_landmarks(self, results, image_shape):
        """
        从检测结果中提取关键点坐标

        Args:
            results: HandLandmarkerResult检测结果对象
            image_shape (tuple): 图像形状 (height, width, channels)

        Returns:
            np.ndarray: 关键点数组，形状为(n_hands, 21, 3)
        """
        landmarks = []
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                hand_landmark = []
                for lm in hand_landmarks:
                    x = lm.x * image_shape[1]
                    y = lm.y * image_shape[0]
                    z = lm.z
                    hand_landmark.append([x, y, z])
                landmarks.append(hand_landmark)
        return np.array(landmarks)

    def draw_landmarks(self, image, results):
        """
        在图像上绘制手部关键点和骨架

        Args:
            image: 输入图像（BGR格式）
            results: HandLandmarkerResult检测结果对象

        Returns:
            绘制了关键点的图像
        """
        if results.hand_landmarks:
            h, w = image.shape[:2]

            for hand_landmarks in results.hand_landmarks:
                # 绘制连接线
                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]

                        start_x = int(start.x * w)
                        start_y = int(start.y * h)
                        end_x = int(end.x * w)
                        end_y = int(end.y * h)

                        cv2.line(image, (start_x, start_y), (end_x, end_y),
                                (0, 255, 0), 2)

                # 绘制关键点
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        return image

    def close(self):
        """关闭检测器，释放资源"""
        self.detector.close()


class PoseDetector:
    """
    姿态关键点检测器（新版Task API）

    使用 MediaPipe Pose 模型检测人体姿态关键点。
    仅提取上半身15个关键点（索引0~14）。
    """

    def __init__(self, min_detection_confidence=0.5):
        """
        初始化姿态检测器

        Args:
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
        """
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # 获取模型文件路径
        models_dir = get_model_path()
        model_path = os.path.join(models_dir, 'pose_landmarker_lite.task')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请运行以下命令下载模型:\n"
                f"python learning/download_models.py"
            )

        # 使用本地模型文件
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(self.options)

        # 上半身姿态关键点连接关系（从共享常量导入）
        self.pose_connections = POSE_CONNECTIONS_UPPER_BODY

    def detect(self, image):
        """
        检测姿态关键点

        Args:
            image: 输入图像（BGR格式）

        Returns:
            PoseLandmarkerResult: 检测结果对象
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        return results

    def get_landmarks(self, results, image_shape):
        """
        从检测结果中提取上半身关键点坐标

        Args:
            results: PoseLandmarkerResult检测结果对象
            image_shape (tuple): 图像形状

        Returns:
            np.ndarray: 关键点数组，形状为(15, 3)
        """
        landmarks = []
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]
            for i in range(15):
                if i < len(pose_landmarks):
                    lm = pose_landmarks[i]
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
            results: PoseLandmarkerResult检测结果对象

        Returns:
            绘制了关键点的图像
        """
        if results.pose_landmarks:
            h, w = image.shape[:2]
            pose_landmarks = results.pose_landmarks[0]

            # 绘制连接线
            for connection in self.pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]

                    start_x = int(start.x * w)
                    start_y = int(start.y * h)
                    end_x = int(end.x * w)
                    end_y = int(end.y * h)

                    cv2.line(image, (start_x, start_y), (end_x, end_y),
                            (0, 255, 0), 2)

            # 绘制关键点
            for i in range(min(15, len(pose_landmarks))):
                lm = pose_landmarks[i]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        return image

    def close(self):
        """关闭检测器，释放资源"""
        self.detector.close()


class HolisticDetector:
    """
    综合检测器（新版Task API）

    同时检测手部和姿态关键点，将两者的数据合并为一个特征向量。
    用于手语数据采集场景。
    """

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5):
        """
        初始化综合检测器

        Args:
            max_num_hands (int, optional): 最大检测手数. Defaults to 2.
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
        """
        self.hand_detector = HandDetector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.pose_detector = PoseDetector(
            min_detection_confidence=min_detection_confidence
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

        hand_landmarks = self.hand_detector.get_landmarks(hand_results, image_shape)
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if len(hand_landmarks) > 0:
            if len(hand_landmarks) >= 1:
                left_hand = hand_landmarks[0]
            if len(hand_landmarks) >= 2:
                right_hand = hand_landmarks[1]

        pose_landmarks = self.pose_detector.get_landmarks(pose_results, image_shape)
        if len(pose_landmarks) == 0:
            pose_landmarks = np.zeros((15, 3))

        combined = np.concatenate([
            left_hand.flatten(),
            right_hand.flatten(),
            pose_landmarks.flatten()
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
