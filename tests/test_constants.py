"""测试 src/constants.py 常量定义"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS_UPPER_BODY,
    POSE_CONNECTIONS_HOLISTIC,
    RAW_HAND_LANDMARKS,
    RAW_HAND_DIMS,
    RAW_HAND_FEATURES,
    RAW_POSE_LANDMARKS,
    RAW_HOLISTIC_FEATURES,
    EXTRACTED_FEATURE_DIMS,
    DEFAULT_SEQUENCE_LENGTH,
)


class TestHandConnections:
    """测试手部骨架连接定义"""

    def test_all_indices_in_range(self):
        """所有连接索引应在 0-20 范围内（21个关键点）"""
        for start, end in HAND_CONNECTIONS:
            assert 0 <= start < 21, f"索引 {start} 超出范围"
            assert 0 <= end < 21, f"索引 {end} 超出范围"

    def test_no_duplicate_connections(self):
        """不应有重复的连接"""
        unique = set()
        for conn in HAND_CONNECTIONS:
            key = tuple(sorted(conn))
            assert key not in unique, f"重复连接: {conn}"
            unique.add(key)


class TestFeatureDimensions:
    """测试特征维度常量的一致性"""

    def test_raw_hand_features(self):
        """RAW_HAND_FEATURES = RAW_HAND_LANDMARKS * RAW_HAND_DIMS"""
        assert RAW_HAND_FEATURES == RAW_HAND_LANDMARKS * RAW_HAND_DIMS

    def test_holistic_features(self):
        """RAW_HOLISTIC_FEATURES = 2 * RAW_HAND_FEATURES + RAW_POSE_LANDMARKS * 3"""
        assert RAW_HOLISTIC_FEATURES == 2 * RAW_HAND_FEATURES + RAW_POSE_LANDMARKS * 3

    def test_extracted_features(self):
        """EXTRACTED_FEATURE_DIMS = RAW_HAND_FEATURES + 4 + 4"""
        assert EXTRACTED_FEATURE_DIMS == RAW_HAND_FEATURES + 4 + 4


class TestSequenceConstants:
    """测试序列常量"""

    def test_default_sequence_length(self):
        assert DEFAULT_SEQUENCE_LENGTH == 30
