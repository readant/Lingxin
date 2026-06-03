"""
数据增强模块 — 关键点序列数据增强

对关键点序列应用随机变换，增加数据多样性，提升模型泛化能力。
所有增强操作均在 (frames, landmarks, coords) 张量上进行。

支持的增强操作：
- 随机平移：对关键点坐标添加微小平移
- 随机缩放：对手部关键点进行缩放（模拟不同距离）
- 时间扭曲：对序列帧进行插值扭曲（模拟速度变化）
- 随机遮挡：随机丢弃部分关键点（模拟遮挡）
- 高斯噪声：对关键点坐标添加噪声（模拟检测误差）

使用示例：
>>> from src.features.augmentation import KeypointAugmenter
>>> augmenter = KeypointAugmenter(p=0.5)
>>> augmented = augmenter(sequence)  # sequence shape: (frames, features)
"""

import numpy as np
from typing import Optional, Tuple


class KeypointAugmenter:
    """
    关键点序列增强器

    应用多种随机增强策略，所有变换保持手部结构的拓扑关系。
    """

    def __init__(
        self,
        p: float = 0.5,
        translate_range: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.01,
        dropout_prob: float = 0.05,
        time_warp_sigma: float = 3.0,
    ):
        """
        Args:
            p: 每个增强操作的应用概率（独立概率）
            translate_range: 平移范围（相对于归一化坐标的比例）
            scale_range: 缩放范围 (min, max)
            noise_std: 高斯噪声标准差
            dropout_prob: 关键点随机丢弃概率
            time_warp_sigma: 时间扭曲平滑度参数
        """
        self.p = p
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.time_warp_sigma = time_warp_sigma

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        对关键点序列应用随机增强

        Args:
            sequence: 输入序列，形状 (frames, features) 或 (frames, landmarks, coords)

        Returns:
            np.ndarray: 增强后的序列，形状与输入相同
        """
        result = sequence.copy().astype(np.float32)
        is_2d = result.ndim == 2

        # 对于 2D 输入 (frames, features)，重排为 (frames, landmarks, coords)
        # 这里假设 features = landmarks * 3，实际上 71 维特征不能直接这样分解
        # 所以我们对 2D 输入只做平移、噪声、时间扭曲
        if is_2d:
            if np.random.random() < self.p:
                result = self._add_noise(result)
            if np.random.random() < self.p:
                result = self._time_warp(result)
            if np.random.random() < self.p:
                result = self._translate(result)
        else:
            # 3D 输入 (frames, landmarks, coords)，可以做所有操作
            if np.random.random() < self.p:
                result = self._translate(result)
            if np.random.random() < self.p:
                result = self._scale(result)
            if np.random.random() < self.p:
                result = self._add_noise(result)
            if np.random.random() < self.p:
                result = self._dropout(result)
            if np.random.random() < self.p:
                result = self._time_warp(result)

        return result

    def _translate(self, sequence: np.ndarray) -> np.ndarray:
        """随机平移"""
        tx = np.random.uniform(-self.translate_range, self.translate_range)
        ty = np.random.uniform(-self.translate_range, self.translate_range)

        if sequence.ndim == 2:
            # (frames, features): 只平移坐标部分（每3个维度中的前2个）
            result = sequence.copy()
            for i in range(0, sequence.shape[1], 3):
                result[:, i] += tx
                result[:, i + 1] += ty
            return result
        else:
            sequence = sequence.copy()
            sequence[..., 0] += tx
            sequence[..., 1] += ty
        return sequence

    def _scale(self, sequence: np.ndarray) -> np.ndarray:
        """随机缩放（3D输入专用：缩放所有坐标）"""
        scale = np.random.uniform(*self.scale_range)
        sequence = sequence.copy()
        sequence[..., :2] *= scale
        return sequence

    def _add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, sequence.shape).astype(np.float32)
        return sequence + noise

    def _dropout(self, sequence: np.ndarray) -> np.ndarray:
        """随机丢弃关键点（3D输入专用）"""
        mask = np.random.random(sequence.shape[1]) > self.dropout_prob
        sequence = sequence.copy()
        sequence[:, ~mask, :] = 0
        return sequence

    def _time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """
        时间扭曲：通过随机插值改变帧的时间分布
        """
        num_frames = sequence.shape[0]
        if num_frames < 3:
            return sequence

        # 对于高维数据，重塑为 2D 处理
        original_shape = sequence.shape
        if sequence.ndim > 2:
            sequence_2d = sequence.reshape(num_frames, -1)
        else:
            sequence_2d = sequence

        # 生成随机控制点
        src = np.arange(num_frames, dtype=np.float32)
        num_control_points = max(2, num_frames // 5)
        control_src = np.linspace(0, num_frames - 1, num_control_points)
        control_dst = control_src + np.random.normal(
            0, self.time_warp_sigma, num_control_points
        )
        control_dst = np.clip(control_dst, 0, num_frames - 1)
        control_dst[0] = 0
        control_dst[-1] = num_frames - 1
        control_dst = np.sort(control_dst)

        # 线性插值重采样
        warped_indices = np.interp(src, control_dst, control_src)

        # 对每个特征维度进行插值
        result = np.zeros_like(sequence_2d)
        for i in range(sequence_2d.shape[1]):
            result[:, i] = np.interp(src, warped_indices, sequence_2d[:, i])

        return result.reshape(original_shape).astype(np.float32)
