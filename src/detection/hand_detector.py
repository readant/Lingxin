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

import os
import sys

# 禁用 MediaPipe 遥测上传（clearcut）
# 必须在 import mediapipe 之前设置，因为 MediaPipe 在模块加载时即初始化遥测线程。
# 在国内网络环境下，clearcut 连接 Google 服务器会超时（Status_ConnectFailed: 12002），
# 后台线程反复尝试连接导致线程争用和 I/O 阻塞，造成视频采集卡顿。
if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel'] = '3'          # 屏蔽 WARNING/ERROR 日志写入 stderr
if 'GLOG_logtostderr' not in os.environ:
    os.environ['GLOG_logtostderr'] = '0'          # 禁止写 stderr
if 'GLOG_alsologtostderr' not in os.environ:
    os.environ['GLOG_alsologtostderr'] = '0'      # 禁止 also-log-to-stderr
if 'MEDIAPIPE_DISABLE_ANALYTICS' not in os.environ:
    os.environ['MEDIAPIPE_DISABLE_ANALYTICS'] = '1'  # 禁用分析数据上报（若支持）

import cv2
import mediapipe as mp
import numpy as np

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

        # 使用本地模型文件（hand_landmarker_lite 已是轻量模型）
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

        # 使用本地模型文件（pose_landmarker_lite 已是轻量模型）
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

    优化：
    - 共享一次 BGR→RGB + mp.Image 创建
    - 姿态降频检测（pose_interval）：姿态变化慢，每 N 帧跑一次即可
    - 使用轻量模型（hand_landmarker_lite, pose_landmarker_lite）
    """

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5,
                 pose_interval=3):
        """
        初始化综合检测器

        Args:
            max_num_hands (int, optional): 最大检测手数. Defaults to 2.
            min_detection_confidence (float, optional): 检测置信度阈值. Defaults to 0.5.
            pose_interval (int, optional): 姿态检测间隔（帧）. 3 表示每 3 帧跑一次.
                                           姿态变化远慢于手部，降频可节省 ~30% 推理时间.
        """
        self.hand_detector = HandDetector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.pose_detector = PoseDetector(
            min_detection_confidence=min_detection_confidence
        )

        # 姿态降频检测
        self.pose_interval = max(1, pose_interval)
        self._frame_count = 0
        self._cached_pose_results = None

    def detect(self, image, rgb_image=None):
        """
        同时检测手部和姿态关键点（姿态检测降频）

        优化：共享一次 BGR→RGB 转换 + mp.Image 创建，
        姿态每 N 帧检测一次，其余帧复用缓存结果。

        Args:
            image: 输入图像（BGR格式），仅在 rgb_image 为 None 时使用
            rgb_image: 可选的预转换 RGB numpy 数组，传入后跳过重复 cv2.cvtColor

        Returns:
            tuple: (hand_results, pose_results)
        """
        if rgb_image is None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 手部关键点：每帧必检测（手部运动快，不能降频）
        hand_results = self.hand_detector.detector.detect(mp_image)

        # 姿态关键点：降频检测（上半身姿态变化慢）
        self._frame_count += 1
        if self._frame_count % self.pose_interval == 0 or self._cached_pose_results is None:
            self._cached_pose_results = self.pose_detector.detector.detect(mp_image)

        return hand_results, self._cached_pose_results

    def get_landmarks(self, results, image_shape):
        """
        提取综合关键点向量（171维）

        输出格式：
        - 前63维: 左手关键点（21个点 × 3坐标）
        - 接下来63维: 右手关键点（21个点 × 3坐标）
        - 最后45维: 姿态关键点（15个点 × 3坐标）

        Args:
            results (tuple): (hand_results, pose_results)
            image_shape (tuple): 图像形状 (height, width, channels)

        Returns:
            np.ndarray: 171维综合特征向量（像素坐标）
        """
        hand_results, pose_results = results
        h, w = image_shape[:2]

        # 通过 MediaPipe handedness 标签区分左右手
        # 不依赖检测器返回顺序，确保数据一致性
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if hand_results.hand_landmarks:
            for i, hand_lms in enumerate(hand_results.hand_landmarks):
                # 获取该只手的左右标签
                hand_label = None
                if (hand_results.handedness and
                        i < len(hand_results.handedness)):
                    category = hand_results.handedness[i][0]
                    hand_label = category.category_name  # "Left" or "Right"

                # 提取关键点坐标
                lm_array = np.array([
                    [lm.x * w, lm.y * h, lm.z]
                    for lm in hand_lms
                ])

                # 按标签分配到对应槽位；无标签时回退到索引顺序
                if hand_label == "Left":
                    left_hand = lm_array
                elif hand_label == "Right":
                    right_hand = lm_array
                elif i == 0:
                    left_hand = lm_array   # 无标签回退
                else:
                    right_hand = lm_array  # 无标签回退

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
