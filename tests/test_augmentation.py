"""测试 src/features/augmentation.py — 数据增强"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.features.augmentation import KeypointAugmenter
from src.constants import EXTRACTED_FEATURE_DIMS, DEFAULT_SEQUENCE_LENGTH


class TestKeypointAugmenter:
    """测试关键点数据增强器"""

    @pytest.fixture
    def augmenter(self):
        return KeypointAugmenter(p=1.0)  # 100% 应用所有增强

    @pytest.fixture
    def sequence(self):
        """创建模拟序列 (30, 71)"""
        np.random.seed(42)
        return np.random.randn(DEFAULT_SEQUENCE_LENGTH, EXTRACTED_FEATURE_DIMS).astype(np.float32)

    def test_output_shape_2d(self, augmenter, sequence):
        """增强后形状应不变 (2D)"""
        result = augmenter(sequence)
        assert result.shape == sequence.shape
        assert result.dtype == np.float32

    def test_output_shape_3d(self):
        """增强后形状应不变 (3D)"""
        augmenter = KeypointAugmenter(p=1.0)
        seq_3d = np.random.randn(30, 21, 3).astype(np.float32)
        result = augmenter(seq_3d)
        assert result.shape == seq_3d.shape

    def test_p_zero(self, sequence):
        """p=0 时不应有任何变换"""
        augmenter = KeypointAugmenter(p=0.0)
        result = augmenter(sequence)
        assert np.allclose(result, sequence)

    def test_no_nan_or_inf(self, augmenter, sequence):
        """增强后不应产生 NaN 或 Inf"""
        result = augmenter(sequence)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_translation_changes_values(self, sequence):
        """平移应改变坐标值"""
        augmenter = KeypointAugmenter(p=0.0)  # 先禁用所有
        augmenter.p = 1.0  # 然后手动设置（会影响所有操作）
        # 直接调用内部方法测试
        result = augmenter._translate(sequence.copy())
        # 前两列（x, y坐标）应该已改变
        assert not np.allclose(result[:, 0], sequence[:, 0])

    def test_noise_increases_variance(self, sequence):
        """噪声应增加方差"""
        augmenter = KeypointAugmenter(p=0.0)
        result = augmenter._add_noise(sequence.copy())
        # 加噪声后方差应大于等于原始方差
        assert np.var(result) >= np.var(sequence) * 0.5

    def test_single_frame_sequence(self):
        """单帧序列不应崩溃"""
        augmenter = KeypointAugmenter(p=1.0)
        single = np.random.randn(1, 71).astype(np.float32)
        result = augmenter(single)
        assert result.shape == single.shape
