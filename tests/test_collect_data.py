"""测试 tools/collect_data.py — 数据完整性端到端验证

核心原则：不测"代码能不能跑"，测"数据对不对"。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import json
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def make_collector(tmp_path, person_id='A', words=None):
    """创建最小化 DataCollector，跳过摄像头/MediaPipe/字体初始化"""
    if words is None:
        words = ['你好', '谢谢', '再见']

    vocab_path = tmp_path / 'vocab.csv'
    vocab_path.write_text(
        'word,category,priority\n' +
        '\n'.join(f'{w},测试,1' for w in words),
        encoding='utf-8'
    )

    with patch('tools.collect_data.HolisticDetector'), \
         patch('tools.collect_data.DataCollector._pre_render_static'):
        from tools.collect_data import DataCollector
        return DataCollector(
            person_id=person_id,
            save_dir=str(tmp_path / 'data'),
            target_samples=30,
            vocab_path=str(vocab_path),
        )


def make_sequence(n_frames=30, seed=42):
    """生成可复现的模拟关键点序列（像素坐标）"""
    rng = np.random.RandomState(seed)
    return [rng.rand(171).astype(np.float32).tolist() for _ in range(n_frames)]


# ═══════════════════════════════════════════════════════════════════════════════
# 端到端：保存 → 加载 → 验证数据完整性
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """保存后加载，验证数据完整性"""

    def test_shape_and_dtype(self, tmp_path):
        """保存的 npy 文件应为 (n_frames, 171) float32"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)
        seq = make_sequence(30)
        c._save_sequence('你好', seq)

        npy_file = list((tmp_path / 'data' / '你好').glob('*.npy'))[0]
        data = np.load(npy_file)
        assert data.dtype == np.float32
        assert data.shape == (30, 171)

    def test_normalization_range(self, tmp_path):
        """归一化后 x,y 坐标应严格在 [0,1] 范围内"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)

        # 构造已知坐标的序列：左手第一个点 x=500, y=300
        seq = []
        for _ in range(20):
            frame = np.zeros(171, dtype=np.float32)
            frame[0] = 500.0   # 左手点0 x
            frame[1] = 300.0   # 左手点0 y
            frame[3] = 100.0   # 左手点1 x
            frame[4] = 400.0   # 左手点1 y
            seq.append(frame.tolist())

        c._save_sequence('你好', seq)

        npy_file = list((tmp_path / 'data' / '你好').glob('*.npy'))[0]
        data = np.load(npy_file)

        # x 坐标: 500/640 ≈ 0.78125, 100/640 ≈ 0.15625
        assert abs(data[0, 0] - 500/640) < 1e-5
        assert abs(data[0, 1] - 300/480) < 1e-5
        assert abs(data[0, 3] - 100/640) < 1e-5
        assert abs(data[0, 4] - 400/480) < 1e-5

        # z 坐标保持原始值（未归一化）
        # z 值是 MediaPipe 原始值，这里构造的都是 0，验证不被修改
        assert data[0, 2] == 0.0

    def test_denormalization_recovers_original(self, tmp_path):
        """反归一化应恢复原始像素坐标"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)

        # 原始像素坐标
        original_x, original_y = 320.0, 240.0
        seq = []
        for _ in range(20):
            frame = np.zeros(171, dtype=np.float32)
            frame[0] = original_x
            frame[1] = original_y
            seq.append(frame.tolist())

        c._save_sequence('你好', seq)

        # 加载并反归一化
        npy_file = list((tmp_path / 'data' / '你好').glob('*.npy'))[0]
        data = np.load(npy_file)
        recovered_x = data[0, 0] * 640
        recovered_y = data[0, 1] * 480

        assert abs(recovered_x - original_x) < 1e-3
        assert abs(recovered_y - original_y) < 1e-3

    def test_metadata_matches_data(self, tmp_path):
        """元信息 JSON 应与 npy 数据一致"""
        c = make_collector(tmp_path, person_id='L')
        c.frame_shape = (480, 640, 3)
        seq = make_sequence(25)
        c._save_sequence('你好', seq)

        npy_file = list((tmp_path / 'data' / '你好').glob('*.npy'))[0]
        meta_file = npy_file.with_name(npy_file.stem.replace('.npy', '') + '_meta.json')
        # 文件名格式: L_001.npy → L_001_meta.json
        meta_file = tmp_path / 'data' / '你好' / 'L_001_meta.json'

        data = np.load(npy_file)
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        assert meta['person_id'] == 'L'
        assert meta['word'] == '你好'
        assert meta['num_frames'] == data.shape[0]
        assert meta['feature_dim'] == data.shape[1]
        assert meta['normalized'] is True
        assert meta['original_resolution']['width'] == 640
        assert meta['original_resolution']['height'] == 480


# ═══════════════════════════════════════════════════════════════════════════════
# 序列长度约束
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceConstraints:
    """验证序列长度约束的实际效果"""

    def test_short_sequence_rejected(self, tmp_path):
        """少于15帧的序列不应产生任何文件"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)
        seq = make_sequence(14)
        success, _ = c._save_sequence('你好', seq)
        assert success is False
        assert len(list((tmp_path / 'data' / '你好').glob('*.npy'))) == 0

    def test_exactly_15_frames_accepted(self, tmp_path):
        """刚好15帧应被接受"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)
        seq = make_sequence(15)
        success, _ = c._save_sequence('你好', seq)
        assert success is True
        data = np.load(list((tmp_path / 'data' / '你好').glob('*.npy'))[0])
        assert data.shape[0] == 15

    def test_long_sequence_truncated_to_150(self, tmp_path):
        """超过150帧应被中心裁剪到150帧"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)
        seq = make_sequence(200)
        c._save_sequence('你好', seq)

        data = np.load(list((tmp_path / 'data' / '你好').glob('*.npy'))[0])
        assert data.shape[0] == 150

    def test_center_crop_keeps_middle(self, tmp_path):
        """中心裁剪应保留中间部分"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)

        # 构造可区分的序列：帧号写入 z 坐标（z 不被归一化）
        seq = []
        for i in range(160):
            frame = np.zeros(171, dtype=np.float32)
            frame[2::3] = float(i)  # z 坐标位置，不会被归一化
            seq.append(frame.tolist())

        c._save_sequence('你好', seq)

        data = np.load(list((tmp_path / 'data' / '你好').glob('*.npy'))[0])
        # 中心裁剪: start = (160-150)//2 = 5, 取 [5:155]
        # 第一帧 z 值应为 5.0，最后一帧应为 154.0
        assert data[0, 2] == pytest.approx(5.0)
        assert data[-1, 2] == pytest.approx(154.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 多人数据隔离
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiPersonIsolation:
    """不同人员的数据不应互相干扰"""

    def test_different_persons_get_separate_files(self, tmp_path):
        """A 和 B 的文件应独立存在"""
        c_a = make_collector(tmp_path, person_id='A')
        c_a.frame_shape = (480, 640, 3)
        seq = make_sequence(20, seed=1)
        c_a._save_sequence('你好', seq)

        c_b = make_collector(tmp_path, person_id='B')
        c_b.frame_shape = (480, 640, 3)
        seq = make_sequence(20, seed=2)
        c_b._save_sequence('你好', seq)

        files = sorted(f.name for f in (tmp_path / 'data' / '你好').glob('*.npy'))
        assert files == ['A_001.npy', 'B_001.npy']

    def test_delete_a_does_not_affect_b(self, tmp_path):
        """删除 A 的样本不应影响 B"""
        c_a = make_collector(tmp_path, person_id='A')
        c_a.frame_shape = (480, 640, 3)
        c_a._save_sequence('你好', make_sequence(20, seed=1))
        c_a._save_sequence('你好', make_sequence(20, seed=2))

        c_b = make_collector(tmp_path, person_id='B')
        c_b.frame_shape = (480, 640, 3)
        c_b._save_sequence('你好', make_sequence(20, seed=3))

        # A 删除最后一个
        c_a._delete_last_sequence('你好')

        files_a = sorted(f.name for f in (tmp_path / 'data' / '你好').glob('A_*.npy'))
        files_b = sorted(f.name for f in (tmp_path / 'data' / '你好').glob('B_*.npy'))
        assert files_a == ['A_001.npy']
        assert files_b == ['B_001.npy']

    def test_counts_are_per_person(self, tmp_path):
        """已录制计数应按人员独立统计"""
        c_a = make_collector(tmp_path, person_id='A')
        c_a.frame_shape = (480, 640, 3)
        c_a._save_sequence('你好', make_sequence(20))
        c_a._save_sequence('你好', make_sequence(20))

        c_b = make_collector(tmp_path, person_id='B')
        c_b.frame_shape = (480, 640, 3)
        c_b._save_sequence('你好', make_sequence(20))

        assert c_a.recorded_counts['你好'] == 2
        assert c_b.recorded_counts['你好'] == 1

    def test_data_values_not_shared(self, tmp_path):
        """A 和 B 保存的数据内容应不同"""
        c_a = make_collector(tmp_path, person_id='A')
        c_a.frame_shape = (480, 640, 3)
        c_a._save_sequence('你好', make_sequence(20, seed=42))

        c_b = make_collector(tmp_path, person_id='B')
        c_b.frame_shape = (480, 640, 3)
        c_b._save_sequence('你好', make_sequence(20, seed=99))

        data_a = np.load(list((tmp_path / 'data' / '你好').glob('A_*.npy'))[0])
        data_b = np.load(list((tmp_path / 'data' / '你好').glob('B_*.npy'))[0])
        assert not np.array_equal(data_a, data_b)


# ═══════════════════════════════════════════════════════════════════════════════
# 删除后序号一致性
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeleteAndReSave:
    """删除后再保存，序号应正确且不覆盖"""

    def test_delete_last_then_save_reuses_index(self, tmp_path):
        """删除最后一个后，新保存的文件会复用该序号（计数器回退）"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)
        c._save_sequence('你好', make_sequence(20))
        c._save_sequence('你好', make_sequence(20))
        c._save_sequence('你好', make_sequence(20))

        c._delete_last_sequence('你好')
        c._save_sequence('你好', make_sequence(20))

        files = sorted(f.name for f in (tmp_path / 'data' / '你好').glob('*.npy'))
        # 删除 A_003 后计数变为 2，新文件序号为 2+1=3，复用了 A_003
        assert files == ['A_001.npy', 'A_002.npy', 'A_003.npy']

    def test_all_files_preserved_after_delete_re_save(self, tmp_path):
        """删除再保存后，所有文件都应完整可读"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)

        # 保存 3 个样本，每个用不同 seed
        for i in range(3):
            c._save_sequence('你好', make_sequence(20, seed=i * 10))

        # 删除第 3 个，再保存一个新样本
        c._delete_last_sequence('你好')
        c._save_sequence('你好', make_sequence(20, seed=99))

        # 所有文件都应可正确加载
        files = sorted((tmp_path / 'data' / '你好').glob('*.npy'))
        assert len(files) == 3
        for f in files:
            data = np.load(f)
            assert data.shape == (20, 171)
            assert data.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# 特征维度一致性
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureConsistency:
    """验证 171 维特征的布局正确性"""

    def test_hand_pose_layout(self, tmp_path):
        """左手63维 + 右手63维 + 姿态45维 = 171维"""
        c = make_collector(tmp_path)
        c.frame_shape = (480, 640, 3)

        # 构造各区域有不同值的序列（只用 z 坐标验证，z 不被归一化）
        seq = []
        for _ in range(20):
            frame = np.zeros(171, dtype=np.float32)
            frame[2:63:3] = 10.0   # 左手 z 坐标 (索引 2,5,...,62)
            frame[65:126:3] = 20.0  # 右手 z 坐标 (索引 65,68,...,125)
            frame[128:171:3] = 30.0 # 姿态 z 坐标 (索引 128,131,...,170)
            seq.append(frame.tolist())

        c._save_sequence('你好', seq)

        data = np.load(list((tmp_path / 'data' / '你好').glob('*.npy'))[0])
        # 左手 21 个 z 值
        assert np.all(data[:, 2:63:3] == 10.0)
        # 右手 21 个 z 值
        assert np.all(data[:, 65:126:3] == 20.0)
        # 姿态 15 个 z 值
        assert np.all(data[:, 128:171:3] == 30.0)
