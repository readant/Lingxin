"""测试 src/config.py — 统一配置管理"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from src.config import ProjectConfig, config


class TestProjectConfig:
    """测试配置类"""

    @pytest.fixture
    def cfg(self):
        return ProjectConfig()

    def test_project_root_exists(self, cfg):
        """项目根目录应存在"""
        assert cfg.project_root.exists()
        assert cfg.project_root.is_dir()

    def test_derived_paths(self, cfg):
        """派生路径应正确"""
        assert cfg.data_dir == cfg.project_root / 'data'
        assert cfg.vocab_path == cfg.data_dir / 'vocab.csv'
        assert cfg.models_dir == cfg.project_root / 'models'

    def test_model_path_classifier(self, cfg):
        """分类器模型路径应为 .pkl"""
        path = cfg.get_model_path('svm')
        assert path.suffix == '.pkl'

    def test_model_path_deep_learning(self, cfg):
        """深度学习模型路径应为 .pth"""
        path = cfg.get_model_path('lstm')
        assert path.suffix == '.pth'

    def test_model_path_unknown(self, cfg):
        """未知模型类型应报错"""
        with pytest.raises(ValueError):
            cfg.get_model_path('unknown_model')

    def test_data_files_classifier(self, cfg):
        """分类器数据文件映射"""
        files = cfg.get_data_files('classifier')
        assert files == {'X': 'X.npy', 'y': 'y.npy'}

    def test_data_files_deep_learning(self, cfg):
        """深度学习数据文件映射"""
        files = cfg.get_data_files('deep_learning')
        assert files == {'X': 'X_sequence.npy', 'y': 'y_sequence.npy'}

    def test_ensure_dirs(self, cfg, tmp_path):
        """ensure_dirs 应创建必要目录"""
        cfg.data_dir = tmp_path / 'test_data'
        cfg.models_dir = tmp_path / 'test_models'
        cfg.ensure_dirs()
        assert cfg.data_dir.exists()
        assert cfg.models_dir.exists()

    def test_global_config(self):
        """全局配置单例应可用"""
        assert config is not None
        assert config.max_sequence_length == 30
        assert config.default_epochs == 50


class TestConfigDefaults:
    """测试配置默认值"""

    def test_sequence_params(self):
        assert config.min_sequence_length == 15
        assert config.max_sequence_length == 30
        assert config.max_raw_sequence_length == 150

    def test_training_params(self):
        assert config.default_learning_rate == 0.001
        assert config.default_batch_size == 32
        assert config.early_stopping_patience == 10

    def test_model_lists(self):
        assert 'svm' in config.classifier_models
        assert 'lstm' in config.deep_learning_models
        assert len(config.all_models) == 5
