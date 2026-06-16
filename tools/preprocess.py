"""
Preprocessor - 数据预处理工具

本文件实现了手语数据的预处理功能，将原始采集的数据转换为训练所需格式。

预处理步骤：
1. 从原始数据目录加载关键点序列（自动解析人员ID）
2. 按人员划分 train/val/test（同一个人不会跨集合）
3. 使用训练集拟合标准化器，再变换所有集合
4. 保存处理后的特征矩阵和标签（分集合）

输出文件（--split-by-person 模式）：
  {output_dir}/
  ├── X_train.npy / y_train.npy       # 训练集
  ├── X_val.npy / y_val.npy           # 验证集
  ├── X_test.npy / y_test.npy         # 测试集
  ├── class_labels.npy                # 类别标签映射
  └── split_info.json                 # 划分详情（哪些人、各集样本数）

使用方法：
  # 所有数据放一起（旧行为，用于快速原型）
  python tools/preprocess.py

  # 自动按人员随机划分
  python tools/preprocess.py --split-by-person

  # 指定哪些人归哪个集合
  python tools/preprocess.py --split-by-person \
      --train-persons J L --val-persons L --test-persons F

  # 按来源区分：视频数据训练，摄像头数据验证
  python tools/preprocess.py --split-by-person \
      --train-persons J --val-persons K --test-persons F
"""

import os
import json
import numpy as np
import argparse
from collections import Counter
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_loader import DataLoader, parse_person_id
from src.config import config
from src.utils.logger import get_logger

logger = get_logger("Preprocessor")


class Preprocessor:
    """
    数据预处理器

    负责将原始采集的手语数据转换为训练所需的格式。
    支持按人员划分数据集，避免数据泄露。

    两种模式：
    - 普通特征：用于 SVM、RF、MLP 等传统机器学习模型
    - 序列特征：用于 LSTM、Transformer 等深度学习模型
    """

    def __init__(self):
        """初始化预处理器"""
        self.loader = DataLoader()
        self.scaler = StandardScaler()

    def preprocess(self, input_dir, output_dir, split_persons=None):
        """
        预处理普通特征数据（传统ML模型用）

        Args:
            input_dir (str): 原始数据目录
            output_dir (str): 输出目录
            split_persons (dict, optional): {'train': [...], 'val': [...], 'test': [...]}
        """
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据（含人员ID）
        X, y, class_labels, person_ids = self.loader.load_data(
            input_dir, return_persons=True
        )

        logger.info(f"加载样本: {len(X)}, 类别: {len(class_labels)}, "
                    f"人员: {len(set(person_ids))}")

        if split_persons:
            self._preprocess_with_split(
                X, y, class_labels, person_ids, split_persons, output_dir,
                prefix=''
            )
        else:
            self._preprocess_simple(X, y, class_labels, output_dir, prefix='')

    def preprocess_sequence(self, input_dir, output_dir, max_length=30,
                           split_persons=None):
        """
        预处理序列特征数据（深度学习模型用）

        Args:
            input_dir (str): 原始数据目录
            output_dir (str): 输出目录
            max_length (int): 最大序列长度
            split_persons (dict, optional): {'train': [...], 'val': [...], 'test': [...]}
        """
        os.makedirs(output_dir, exist_ok=True)

        # 加载序列数据（含人员ID）
        X, y, class_labels, person_ids = self.loader.load_sequence_data(
            input_dir, max_length, return_persons=True
        )

        logger.info(f"加载序列样本: {len(X)}, 类别: {len(class_labels)}, "
                    f"人员: {len(set(person_ids))}")

        if split_persons:
            self._preprocess_with_split(
                X, y, class_labels, person_ids, split_persons, output_dir,
                prefix='sequence_'
            )
        else:
            self._preprocess_simple(X, y, class_labels, output_dir, prefix='sequence_')

    def _preprocess_simple(self, X, y, class_labels, output_dir, prefix=''):
        """简单预处理：所有数据一起标准化（旧行为，不做划分）"""
        import joblib
        # 对于序列数据(3D)，需要reshape为2D进行标准化，再reshape回来
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(-1, n_features)
            X_scaled_2d = self.scaler.fit_transform(X_2d)
            X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_features)
        else:
            X_scaled = self.scaler.fit_transform(X)

        np.save(os.path.join(output_dir, f'{prefix}X.npy'), X_scaled)
        np.save(os.path.join(output_dir, f'{prefix}y.npy'), y)
        np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)

        # 保存 scaler 供推理时使用
        scaler_path = os.path.join(output_dir, f'{prefix}scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"保存到 {output_dir}: {prefix}X.npy (shape={X_scaled.shape})")
        logger.info(f"  类别: {len(class_labels)}, 无数据划分")

    def _preprocess_with_split(self, X, y, class_labels, person_ids,
                                split_persons, output_dir, prefix=''):
        """按人员划分预处理：scaler 仅拟合训练集"""
        # 1. 按人员划分
        splits = self.loader.split_by_person(
            X, y, person_ids,
            train_persons=split_persons.get('train', []),
            val_persons=split_persons.get('val', []),
            test_persons=split_persons.get('test', []),
        )

        # 2. 打印划分信息
        person_dist = Counter(person_ids)
        logger.info("=" * 60)
        logger.info("数据划分 (按人员隔离)")
        logger.info("=" * 60)

        # 统计各集合人员分布
        split_info = {
            'mode': 'split_by_person',
            'total_samples': int(len(X)),
            'total_persons': int(len(set(person_ids))),
            'num_classes': int(len(class_labels)),
            'person_distribution': {p: int(c) for p, c in person_dist.items()},
        }

        for split_name in ['train', 'val', 'test']:
            split_data = splits.get(split_name)
            if split_data is None:
                split_info[f'{split_name}_samples'] = 0
                logger.info(f"  {split_name:5s}: (empty)")
                continue

            X_s, y_s = split_data
            if len(X_s) == 0:
                split_info[f'{split_name}_samples'] = 0
                logger.info(f"  {split_name:5s}: (empty)")
                continue

            # 找出该集合包含的人员
            target_persons = set(split_persons.get(split_name, []))
            persons_in_split = [p for p in target_persons
                              if p in set(person_ids)]

            label_counts = Counter(y_s.tolist())
            split_info[f'{split_name}_samples'] = int(len(X_s))
            split_info[f'{split_name}_persons'] = sorted(persons_in_split)

            # 每个词汇的样本数
            word_counts = {}
            for label_idx, count in sorted(label_counts.items()):
                word = [k for k, v in class_labels.items() if v == label_idx][0]
                word_counts[word] = count
            split_info[f'{split_name}_per_word'] = word_counts

            logger.info(f"  {split_name:5s}: {len(X_s)} samples, "
                      f"{len(persons_in_split)} persons "
                      f"({', '.join(sorted(persons_in_split))})")
            for word, count in sorted(word_counts.items()):
                logger.info(f"    {word}: {count}")

        # 3. 仅在训练集上拟合 scaler
        X_train = splits['train'][0]
        self.scaler.fit(X_train)

        # 4. 分别变换各集并保存
        for split_name in ['train', 'val', 'test']:
            split_data = splits.get(split_name)
            if split_data is not None and len(split_data[0]) > 0:
                X_s, y_s = split_data
                X_scaled = self.scaler.transform(X_s)
                np.save(os.path.join(output_dir, f'{prefix}X_{split_name}.npy'), X_scaled)
                np.save(os.path.join(output_dir, f'{prefix}y_{split_name}.npy'), y_s)
                logger.info(f"保存: {prefix}X_{split_name}.npy (shape={X_scaled.shape})")

        # 5. 保存类别标签
        np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)

        # 6. 保存划分信息（可追溯）
        with open(os.path.join(output_dir, 'split_info.json'), 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

        logger.info("=" * 60)
        logger.info(f"数据已保存到: {output_dir}")
        logger.info(f"类别: {len(class_labels)}")
        logger.info(f"划分详情: {output_dir}/split_info.json")

    def print_person_summary(self, input_dir):
        """
        仅打印数据摘要（不加载完整数据），方便查看各人员分布

        Args:
            input_dir (str): 数据目录
        """
        from collections import defaultdict

        person_word = defaultdict(lambda: defaultdict(int))
        total_persons = set()

        for word_dir in sorted(os.listdir(input_dir)):
            word_path = os.path.join(input_dir, word_dir)
            if not os.path.isdir(word_path):
                continue
            for fn in sorted(os.listdir(word_path)):
                if fn.endswith('.npy') and not fn.endswith('_meta.npy'):
                    pid = parse_person_id(fn)
                    person_word[pid][word_dir] += 1
                    total_persons.add(pid)

        logger.info("=" * 60)
        logger.info("数据人员分布摘要")
        logger.info("=" * 60)

        total_samples = 0
        for pid in sorted(total_persons):
            count = sum(person_word[pid].values())
            total_samples += count
            words_str = ', '.join(
                f'{w}:{c}' for w, c in sorted(person_word[pid].items())
            )
            logger.info(f"  {pid}: {count} samples ({words_str})")

        logger.info(f"\n总计: {len(total_persons)} 人, {total_samples} 个样本")
        logger.info("=" * 60)


def parse_person_list(arg):
    """解析命令行的人员列表参数"""
    if arg is None:
        return None
    return [p.strip() for p in arg.split()]


def main():
    parser = argparse.ArgumentParser(
        description='手语数据预处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 所有数据放一起（快速验证用）
  python tools/preprocess.py

  # 自动按人员随机划分
  python tools/preprocess.py --split-by-person

  # 指定人员划分（按你当前的场景）
  python tools/preprocess.py --split-by-person \\
      --train-persons J L --test-persons F

  # 只查看数据分布，不处理
  python tools/preprocess.py --summary
        """
    )
    parser.add_argument('--input-dir', default=str(config.raw_data_dir),
                       help=f'原始数据目录 (默认: {config.raw_data_dir})')
    parser.add_argument('--output-dir', default=str(config.processed_data_dir),
                       help=f'输出目录 (默认: {config.processed_data_dir})')
    parser.add_argument('--max-length', type=int, default=30,
                       help='最大序列长度 (默认: 30)')

    # 划分模式
    parser.add_argument('--split-by-person', action='store_true',
                       help='启用按人员划分（同一人的数据不会跨集合）')
    parser.add_argument('--train-persons', type=str, default=None,
                       help='训练集人员ID列表 (空格分隔)')
    parser.add_argument('--val-persons', type=str, default=None,
                       help='验证集人员ID列表 (空格分隔)')
    parser.add_argument('--test-persons', type=str, default=None,
                       help='测试集人员ID列表 (空格分隔)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='自动划分时的验证集人员比例 (默认: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='自动划分时的测试集人员比例 (默认: 0.15)')

    # 其他
    parser.add_argument('--summary', action='store_true',
                       help='仅打印数据摘要，不进行预处理')

    args = parser.parse_args()

    preprocessor = Preprocessor()

    # 仅查看摘要
    if args.summary:
        preprocessor.print_person_summary(args.input_dir)
        return

    logger.info("=" * 60)
    logger.info("聆心手语数据预处理")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # 确定划分方案
    split_persons = None

    if args.split_by_person:
        if args.train_persons:
            # 手动指定
            split_persons = {
                'train': parse_person_list(args.train_persons),
                'val': parse_person_list(args.val_persons) or [],
                'test': parse_person_list(args.test_persons) or [],
            }
            logger.info(f"手动划分: train={split_persons['train']}, "
                       f"val={split_persons['val']}, test={split_persons['test']}")
        else:
            # 自动随机划分：先加载数据获取所有人员
            _, _, _, person_ids = preprocessor.loader.load_data(
                args.input_dir, return_persons=True
            )
            split_persons = preprocessor.loader.auto_split_persons(
                person_ids, val_ratio=args.val_ratio, test_ratio=args.test_ratio
            )
            logger.info(f"自动划分: train={split_persons['train']}, "
                       f"val={split_persons['val']}, test={split_persons['test']}")

    # 执行预处理
    logger.info("")
    logger.info("预处理普通特征数据...")
    preprocessor.preprocess(args.input_dir, args.output_dir, split_persons)

    logger.info("")
    logger.info("预处理序列特征数据...")
    preprocessor.preprocess_sequence(
        args.input_dir, args.output_dir, args.max_length, split_persons
    )

    logger.info("")
    logger.info("全部完成！")


if __name__ == '__main__':
    main()
