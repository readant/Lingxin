"""
TrainRunner - 训练入口

本文件是整个项目的训练入口，提供统一的命令行接口用于训练不同类型的模型。

功能特性：
- 支持多种模型类型：SVM、RF、MLP、LSTM、Transformer
- 自动检测预划分数据（X_train/test/val），或使用全量数据
- 用验证集做早停，测试集做最终评估

使用方法：
  # 使用预划分数据（推荐）
  python tools/train.py --model lstm

  # 使用全量数据自动划分
  python tools/train.py --model svm

  # 指定数据目录
  python tools/train.py --model lstm --data data/processed/csl_isolated
"""

import os
import json
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import Trainer
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.config import config
from src.utils.logger import get_logger

logger = get_logger("TrainRunner")


class TrainRunner:
    """
    训练运行器

    负责加载数据、调用训练方法、保存模型、最终评估。
    自动检测数据是否已按 train/val/test 预划分。
    """

    MODEL_MAP = {
        'svm': {'type': 'classifier', 'model_class': None},
        'rf': {'type': 'classifier', 'model_class': None},
        'mlp': {'type': 'classifier', 'model_class': None},
        'lstm': {'type': 'deep_learning', 'model_class': LSTMModel},
        'transformer': {'type': 'deep_learning', 'model_class': TransformerModel}
    }

    def __init__(self):
        pass

    def _load_split_data(self, data_dir, model_category):
        """
        加载数据，自动检测是否预划分

        Returns:
            dict: {
                'X_train', 'y_train',  # 必有
                'X_val', 'y_val',      # 可能为 None
                'X_test', 'y_test',    # 可能为 None
                'class_labels',
            }
        """
        prefix = '' if model_category == 'classifier' else 'sequence_'

        # 检测预划分文件
        train_path = os.path.join(data_dir, f'{prefix}X_train.npy')
        if os.path.exists(train_path):
            logger.info("检测到预划分数据（按人员隔离）")

            data = {}
            for split in ['train', 'val', 'test']:
                x_path = os.path.join(data_dir, f'{prefix}X_{split}.npy')
                y_path = os.path.join(data_dir, f'{prefix}y_{split}.npy')
                if os.path.exists(x_path) and os.path.exists(y_path):
                    data[f'X_{split}'] = np.load(x_path)
                    data[f'y_{split}'] = np.load(y_path)
                    logger.info(f"  {split}: {data[f'X_{split}'].shape[0]} samples")
                else:
                    data[f'X_{split}'] = None
                    data[f'y_{split}'] = None

            data['class_labels'] = np.load(
                os.path.join(data_dir, 'class_labels.npy'), allow_pickle=True
            ).item()

            # 打印划分详情（如果有）
            split_info_path = os.path.join(data_dir, 'split_info.json')
            if os.path.exists(split_info_path):
                with open(split_info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                logger.info(f"  划分模式: {info.get('mode', 'unknown')}")
                logger.info(f"  总人数: {info.get('total_persons', '?')}")
        else:
            # 旧格式：全量数据
            data_files = config.get_data_files(model_category)
            X = np.load(os.path.join(data_dir, data_files['X']))
            y = np.load(os.path.join(data_dir, data_files['y']))
            class_labels = np.load(
                os.path.join(data_dir, 'class_labels.npy'), allow_pickle=True
            ).item()

            data = {
                'X_train': X,
                'y_train': y,
                'X_val': None,
                'y_val': None,
                'X_test': None,
                'y_test': None,
                'class_labels': class_labels,
            }
            logger.info(f"加载全量数据: {X.shape[0]} samples (未预划分，训练时自动分割)")

        return data

    def run(self, data_dir=None, model_type='svm', save_dir=None, **kwargs):
        """执行训练流程"""
        if data_dir is None:
            data_dir = str(config.processed_data_dir)
        if save_dir is None:
            save_dir = str(config.models_dir)

        os.makedirs(save_dir, exist_ok=True)

        if model_type not in self.MODEL_MAP:
            valid_types = ", ".join(self.MODEL_MAP.keys())
            logger.error(f'未知模型类型: {model_type}，可选: {valid_types}')
            return

        model_info = self.MODEL_MAP[model_type]
        category = model_info['type']

        # 加载数据
        data = self._load_split_data(data_dir, category)

        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data.get('X_val')
        y_val = data.get('y_val')
        X_test = data.get('X_test')
        y_test = data.get('y_test')

        logger.info(f'训练集: {X_train.shape[0]} samples')
        if X_val is not None:
            logger.info(f'验证集: {X_val.shape[0]} samples')
        if X_test is not None:
            logger.info(f'测试集: {X_test.shape[0]} samples')
        logger.info(f'类别数: {len(data["class_labels"])}')

        # 训练
        if category == 'classifier':
            logger.info(f'正在训练 {model_type} 分类器...')
            model_data, train_acc = Trainer.train_classifier(
                X_train, y_train, model_type, test_size=0.2
            )
            model_path = config.get_model_path(model_type)
            Trainer.save_classifier(model_data, str(model_path))
            logger.info(f'模型已保存: {model_path}')

            # 分类器在测试集上的评估
            if X_test is not None:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_t_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                y_pred = model_data['model'].predict(X_test_scaled)
                test_acc = np.mean(y_pred == y_test)
                logger.info(f'测试集准确率: {test_acc:.4f} ({len(y_test)} samples)')

        else:
            model_class = model_info['model_class']
            save_best = str(config.get_model_path(model_type))

            logger.info(f'正在训练 {model_type} 模型...')
            model, val_acc = Trainer.train_deep_learning(
                X_train, y_train, model_class,
                X_val=X_val, y_val=y_val,        # ← 预划分的验证集
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32),
                lr=kwargs.get('lr', 0.001),
                device=kwargs.get('device', 'auto'),
                early_stopping_patience=kwargs.get('patience', 10),
                save_best_path=save_best,
            )

            # 测试集最终评估
            if X_test is not None:
                import torch
                test_acc = Trainer.evaluate_model(model, X_test, y_test)
                logger.info(f'测试集准确率: {test_acc:.4f} ({len(y_test)} samples)')

            logger.info(f'验证集准确率: {val_acc:.4f}')
            logger.info(f'模型已保存: {save_best}')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='聆心手语识别 — 模型训练')
    parser.add_argument('--model', type=str, required=True,
                       choices=['svm', 'rf', 'mlp', 'lstm', 'transformer'],
                       help='模型类型')
    parser.add_argument('--data', type=str, default=None,
                       help='数据目录 (默认: config.processed_data_dir)')
    parser.add_argument('--save', type=str, default=None,
                       help='模型保存目录 (默认: config.models_dir)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停容忍轮数')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'], help='训练设备')

    args = parser.parse_args()

    runner = TrainRunner()
    runner.run(
        data_dir=args.data,
        model_type=args.model,
        save_dir=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )


if __name__ == '__main__':
    main()
