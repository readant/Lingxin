"""测试 src/training/trainer.py — Trainer"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import tempfile
import os
from src.training.trainer import Trainer


class TestTrainerClassifier:
    """测试传统机器学习分类器训练"""

    @pytest.fixture
    def sample_data(self):
        """生成简单的分类数据"""
        np.random.seed(42)
        X = np.random.randn(100, 71).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50)
        return X, y

    def test_train_svm(self, sample_data):
        """SVM训练应返回模型和准确率"""
        X, y = sample_data
        model_data, accuracy = Trainer.train_classifier(X, y, model_type='svm')

        assert 'model' in model_data
        assert 'scaler' in model_data
        assert model_data['type'] == 'svm'
        assert 0 <= accuracy <= 1

    def test_train_rf(self, sample_data):
        """随机森林训练应返回模型和准确率"""
        X, y = sample_data
        model_data, accuracy = Trainer.train_classifier(X, y, model_type='rf')

        assert model_data['type'] == 'rf'
        assert 0 <= accuracy <= 1

    def test_train_mlp(self, sample_data):
        """MLP训练应返回模型和准确率"""
        X, y = sample_data
        model_data, accuracy = Trainer.train_classifier(X, y, model_type='mlp')

        assert model_data['type'] == 'mlp'
        assert 0 <= accuracy <= 1

    def test_invalid_model_type(self, sample_data):
        """无效模型类型应抛出ValueError"""
        X, y = sample_data
        with pytest.raises(ValueError):
            Trainer.train_classifier(X, y, model_type='invalid')

    def test_save_load_classifier(self, sample_data, tmp_path):
        """分类器保存和加载应保持一致"""
        X, y = sample_data
        model_data, _ = Trainer.train_classifier(X, y, model_type='svm')

        save_path = str(tmp_path / 'test_model.pkl')
        Trainer.save_classifier(model_data, save_path)

        loaded = Trainer.load_classifier(save_path)
        assert loaded['type'] == model_data['type']
        assert hasattr(loaded['model'], 'predict')


class TestTrainerDeepLearning:
    """测试深度学习模型训练"""

    @pytest.fixture
    def sequence_data(self):
        """生成序列分类数据"""
        np.random.seed(42)
        n_samples = 50
        seq_len = 30
        n_features = 71
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y = np.array([0] * 25 + [1] * 25)
        return X, y

    def test_train_lstm(self, sequence_data):
        """LSTM训练应返回模型和准确率"""
        from src.models.lstm_model import LSTMModel

        X, y = sequence_data
        model, accuracy = Trainer.train_deep_learning(
            X, y, LSTMModel, epochs=2, batch_size=16
        )

        assert hasattr(model, 'forward')
        assert 0 <= accuracy <= 1

    def test_train_transformer(self, sequence_data):
        """Transformer训练应返回模型和准确率"""
        from src.models.transformer_model import TransformerModel

        X, y = sequence_data
        model, accuracy = Trainer.train_deep_learning(
            X, y, TransformerModel, epochs=2, batch_size=16
        )

        assert hasattr(model, 'forward')
        assert 0 <= accuracy <= 1

    def test_evaluate_model(self, sequence_data):
        """模型评估应返回准确率"""
        from src.models.lstm_model import LSTMModel

        X, y = sequence_data
        model, _ = Trainer.train_deep_learning(
            X, y, LSTMModel, epochs=2, batch_size=16
        )

        accuracy = Trainer.evaluate_model(model, X[:10], y[:10])
        assert 0 <= accuracy <= 1
