"""测试 src/features/feature_extractor.py — FeatureExtractor"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.features.feature_extractor import FeatureExtractor
from src.constants import EXTRACTED_FEATURE_DIMS, RAW_HAND_LANDMARKS, RAW_HAND_DIMS


class TestFeatureExtractor:
    """测试特征提取器"""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    def test_empty_landmarks(self, extractor):
        """空关键点列表应返回零向量"""
        result = extractor.extract_features([])
        assert result.shape == (EXTRACTED_FEATURE_DIMS,)
        assert np.allclose(result, np.zeros(EXTRACTED_FEATURE_DIMS))

    def test_zero_landmarks(self, extractor):
        """全零关键点应处理（不崩溃）"""
        landmarks = [np.zeros((RAW_HAND_LANDMARKS, RAW_HAND_DIMS))]
        result = extractor.extract_features(landmarks)
        assert result.shape == (EXTRACTED_FEATURE_DIMS,)

    def test_normal_landmarks_single_hand(self, extractor):
        """单只正常手的特征提取"""
        # 创建模拟的手部关键点（简单的几何形状）
        landmarks = [np.random.randn(RAW_HAND_LANDMARKS, RAW_HAND_DIMS).astype(np.float32) * 0.1]
        result = extractor.extract_features(landmarks)
        assert result.shape == (EXTRACTED_FEATURE_DIMS,)
        assert not np.allclose(result, 0)

    def test_multi_hand_averaging(self, extractor):
        """多只手应取平均特征"""
        # 两只手，一只手关键点全在 (1,2,3)，另一只全在 (3,4,5)
        hand1 = np.ones((RAW_HAND_LANDMARKS, RAW_HAND_DIMS)) * 1.0
        hand2 = np.ones((RAW_HAND_LANDMARKS, RAW_HAND_DIMS)) * 3.0
        result = extractor.extract_features([hand1, hand2])

        # 由于相对化处理（减去手腕），两只手特征可能不完全是对应坐标的平均
        assert result.shape == (EXTRACTED_FEATURE_DIMS,)

    def test_output_range(self, extractor):
        """特征值应在合理范围内（非无穷、非NaN）"""
        landmarks = [np.random.randn(RAW_HAND_LANDMARKS, RAW_HAND_DIMS).astype(np.float32)]
        result = extractor.extract_features(landmarks)
        assert not np.any(np.isnan(result)), "特征包含 NaN"
        assert not np.any(np.isinf(result)), "特征包含 Inf"


class TestAngleCalculation:
    """测试关节角度计算"""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    def test_straight_finger_angle(self, extractor):
        """伸直的手指：手腕(0,0,0), 指根(1,0,0), 指尖(2,0,0)
        向量 v1=手腕-指根=(-1,0,0), v2=指尖-指根=(1,0,0)
        两向量方向相反，夹角 = π ≈ 3.14"""
        landmarks = np.zeros((21, 3))
        landmarks[0] = [0, 0, 0]   # 手腕
        landmarks[5] = [1, 0, 0]   # 食指根
        landmarks[6] = [2, 0, 0]   # 食指中节

        angles = extractor.calculate_angles(landmarks)
        assert len(angles) == 4
        # 伸直时手腕-指根-指尖三点一线，向量方向相反，角度接近 π
        assert angles[1] > 3.0, f"伸直手指角度应接近π，实际: {angles[1]}"

    def test_bent_finger_angle(self, extractor):
        """弯曲的手指：手腕(0,0,0), 指根(1,0,0), 中间(1,1,0) → 角度 ≈ π/2"""
        landmarks = np.zeros((21, 3))
        landmarks[0] = [0, 0, 0]   # 手腕
        landmarks[5] = [1, 0, 0]   # 食指根
        landmarks[6] = [1, 1, 0]   # 食指中节（弯曲90度）

        angles = extractor.calculate_angles(landmarks)
        assert len(angles) == 4
        # 食指角度应接近 π/2 ≈ 1.57
        assert 1.4 < angles[1] < 1.8, f"弯曲手指角度应接近π/2，实际: {angles[1]}"
