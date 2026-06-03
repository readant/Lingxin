"""
VideoCollector - 从视频文件批量采集手语关键点数据

从已有视频文件中自动提取手语关键点序列，适合批量数据采集。
与 tools/collect_data.py（摄像头实时采集）互补。

工作流程：
1. 扫描视频目录，解析文件名获取词汇和人员信息
2. 逐帧读取视频，使用 MediaPipe Holistic 检测关键点
3. 将完整视频的关键点序列保存为 .npy 文件

文件名约定（支持三种格式）：
  {word}_{person_id}_{index}.mp4     → 你好_A_001.mp4
  {word}_{person_id}.mp4             → 你好_A.mp4
  {word}.mp4                         → 你好.mp4（需指定 --person-id）

输出格式（与摄像头采集一致）：
  - {person_id}_{序号:03d}.npy       → 171维关键点序列 (n_frames, 171)
  - {person_id}_{序号:03d}_meta.json → 元信息

使用示例：
  # 基本用法
  python tools/collect_from_video.py --input-dir data/videos/

  # 指定人员ID（覆盖文件名推断）
  python tools/collect_from_video.py --input-dir data/videos/ --person-id K

  # 跳帧 + 最低置信度过滤
  python tools/collect_from_video.py --input-dir data/videos/ --frame-skip 2 --min-confidence 0.6

  # 指定输出目录
  python tools/collect_from_video.py --input-dir data/videos/ --output-dir data/raw/collected
"""

import cv2
import numpy as np
import os
import sys
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.hand_detector import HolisticDetector
from src.config import config
from src.utils.logger import get_logger

logger = get_logger("VideoCollector")

# 文件名解析正则
# 格式: {word}_{person_id}_{index}.mp4  或  {word}_{person_id}.mp4
FILENAME_RE = re.compile(
    r'^(.+?)_([A-Za-z0-9]+)(?:_(\d+))?\.(mp4|avi|mov|mkv|webm)$',
    re.IGNORECASE
)


def parse_filename(filepath):
    """
    从文件名解析词汇、人员ID、序号

    支持的格式:
      {word}_{person}_{index}.mp4  → ('你好', 'A', 1)
      {word}_{person}.mp4          → ('你好', 'A', None)
      {word}.mp4                   → ('你好', None, None)

    Args:
        filepath (str or Path): 视频文件路径

    Returns:
        dict: {'word': str|None, 'person_id': str|None, 'index': int|None, 'ext': str}
    """
    filename = Path(filepath).name
    match = FILENAME_RE.match(filename)

    if match:
        return {
            'word': match.group(1),
            'person_id': match.group(2),
            'index': int(match.group(3)) if match.group(3) else None,
            'ext': match.group(4),
        }
    else:
        # 纯词汇名: {word}.mp4
        name_no_ext = Path(filepath).stem
        return {
            'word': name_no_ext,
            'person_id': None,
            'index': None,
            'ext': Path(filepath).suffix.lstrip('.'),
        }


class VideoCollector:
    """
    视频文件批量采集器

    从视频目录中读取手语视频，逐帧提取关键点序列并保存。

    Attributes:
        detector: HolisticDetector 实例
        vocab_df: 词汇表 DataFrame
        min_confidence: 最低检测置信度阈值
        frame_skip: 跳帧间隔（每隔N帧采样一次）
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        frame_skip: int = 1,
        person_id: str = None,
        vocab_path: str = None,
    ):
        """
        Args:
            min_confidence: MediaPipe 检测最低置信度
            frame_skip: 跳帧间隔（1=全部帧，2=每隔1帧，3=每隔2帧...）
            person_id: 默认人员ID（文件名中未包含时使用）
            vocab_path: 词汇表路径（默认使用 config.vocab_path）
        """
        self.detector = HolisticDetector(
            min_detection_confidence=min_confidence
        )
        self.min_confidence = min_confidence
        self.frame_skip = max(1, frame_skip)
        self.default_person_id = person_id

        # 加载词汇表
        if vocab_path:
            vocab_path = Path(vocab_path)
        else:
            vocab_path = config.vocab_path
        if vocab_path.exists():
            self.vocab_df = pd.read_csv(str(vocab_path))
            self.valid_words = set(self.vocab_df['word'].tolist())
            logger.info(f"词汇表: {vocab_path} ({len(self.valid_words)} 个词)")
        else:
            logger.warning(f"词汇表不存在: {vocab_path}，将不校验词汇")
            self.vocab_df = None
            self.valid_words = set()

        # 统计
        self.stats = {
            'total_videos': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_frames': 0,
            'total_valid_frames': 0,
        }

    def _get_next_index(self, word_dir, person_id):
        """
        获取下一个样本序号（避免覆盖已有文件）

        Args:
            word_dir (str): 词汇数据目录
            person_id (str): 人员ID

        Returns:
            int: 下一个可用序号
        """
        if not os.path.exists(word_dir):
            return 1
        existing = [
            f for f in os.listdir(word_dir)
            if f.startswith(person_id) and f.endswith('.npy')
        ]
        return len(existing) + 1

    def _get_category(self, word):
        """查找词汇所属类别"""
        if self.vocab_df is not None:
            rows = self.vocab_df[self.vocab_df['word'] == word]
            if len(rows) > 0:
                return rows['category'].values[0]
        return ''

    def process_video(self, video_path, output_dir, person_id=None, word=None):
        """
        处理单个视频文件

        Args:
            video_path (str): 视频文件路径
            output_dir (str): 数据保存根目录
            person_id (str, optional): 人员ID
            word (str, optional): 词汇名

        Returns:
            bool: 是否成功
        """
        video_path = str(video_path)
        filename = Path(video_path).name

        # 解析文件名
        info = parse_filename(video_path)
        word = word or info['word']
        pid = person_id or info['person_id'] or self.default_person_id

        if not word:
            logger.warning(f"跳过（无法确定词汇）: {filename}")
            self.stats['skipped'] += 1
            return False

        if not pid:
            logger.warning(f"跳过（无法确定人员ID）: {filename}，请通过 --person-id 指定")
            self.stats['skipped'] += 1
            return False

        # 校验词汇（如果词汇表可用）
        if self.valid_words and word not in self.valid_words:
            logger.warning(f"跳过（词汇不在词汇表中）: {word}")
            self.stats['skipped'] += 1
            return False

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            self.stats['failed'] += 1
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"处理: {filename} → 词={word}, 人={pid}, "
                    f"帧数={total_frames}, FPS={fps:.1f}")

        # 逐帧提取关键点
        landmarks_sequence = []
        frame_idx = 0
        valid_frame_count = 0
        hand_lost_count = 0

        pbar = tqdm(total=total_frames, desc=f"  {filename}", unit="f", leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 跳帧
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            try:
                results = self.detector.detect(frame)
                landmarks = self.detector.get_landmarks(results, frame.shape)
            except Exception as e:
                logger.debug(f"帧 {frame_idx} 检测失败: {e}")
                frame_idx += 1
                pbar.update(1)
                continue

            # 检查是否检测到手部
            has_hand = not np.all(landmarks[:126] == 0)
            if has_hand:
                landmarks_sequence.append(landmarks)
                valid_frame_count += 1
                hand_lost_count = 0
            else:
                hand_lost_count += 1
                # 允许短暂丢失（连续10帧无手才跳过）
                if hand_lost_count > 10 and len(landmarks_sequence) > 0:
                    # 恢复后继续追加，不截断
                    pass

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        self.stats['total_frames'] += frame_idx
        self.stats['total_valid_frames'] += valid_frame_count

        # 检查序列长度
        min_frames = config.min_sequence_length
        if len(landmarks_sequence) < min_frames:
            logger.warning(f"跳过（有效帧不足）: {filename} "
                          f"({valid_frame_count}/{frame_idx} 帧有效, 最少需要 {min_frames} 帧)")
            self.stats['skipped'] += 1
            return False

        # 保存
        word_dir = os.path.join(output_dir, word)
        os.makedirs(word_dir, exist_ok=True)

        idx = self._get_next_index(word_dir, pid)
        sequence_array = np.array(landmarks_sequence)

        file_name = f'{pid}_{idx:03d}.npy'
        save_path = os.path.join(word_dir, file_name)
        np.save(save_path, sequence_array)

        # 保存元信息
        metadata = {
            'person_id': pid,
            'word': word,
            'category': self._get_category(word),
            'num_frames': len(sequence_array),
            'feature_dim': sequence_array.shape[1] if sequence_array.ndim > 1 else 0,
            'timestamp': datetime.now().isoformat(),
            'file_name': file_name,
            'source_video': filename,
            'source_fps': round(fps, 1),
            'valid_ratio': round(valid_frame_count / max(frame_idx, 1), 3),
        }
        meta_path = save_path.replace('.npy', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"保存: {save_path} (帧数={len(sequence_array)}, "
                   f"有效率={valid_frame_count}/{frame_idx})")
        self.stats['processed'] += 1
        return True

    def process_directory(self, input_dir, output_dir):
        """
        批量处理视频目录

        Args:
            input_dir (str): 视频文件目录
            output_dir (str): 数据保存根目录
        """
        input_dir = str(input_dir)
        if not os.path.isdir(input_dir):
            logger.error(f"目录不存在: {input_dir}")
            return

        # 收集视频文件
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = []
        for f in sorted(os.listdir(input_dir)):
            fpath = os.path.join(input_dir, f)
            if os.path.isfile(fpath):
                _, ext = os.path.splitext(f)
                if ext.lower() in video_exts:
                    video_files.append(fpath)

        if not video_files:
            logger.warning(f"目录中未找到视频文件: {input_dir}")
            logger.info(f"支持的格式: {', '.join(video_exts)}")
            return

        self.stats['total_videos'] = len(video_files)
        logger.info(f"找到 {len(video_files)} 个视频文件")
        logger.info(f"跳帧间隔: {self.frame_skip}（每 {self.frame_skip} 帧采样1帧）")
        logger.info(f"输出目录: {output_dir}")
        logger.info("=" * 60)

        for video_path in video_files:
            self.process_video(video_path, output_dir)

        self._print_summary()

    def _print_summary(self):
        """打印采集汇总"""
        s = self.stats
        logger.info("\n" + "=" * 60)
        logger.info("采集汇总")
        logger.info("=" * 60)
        logger.info(f"  视频总数:     {s['total_videos']}")
        logger.info(f"  成功处理:     {s['processed']}")
        logger.info(f"  跳过:         {s['skipped']}")
        logger.info(f"  失败:         {s['failed']}")
        logger.info(f"  总帧数:       {s['total_frames']}")
        logger.info(f"  有效帧数:     {s['total_valid_frames']}")
        logger.info(f"  整体有效率:   {s['total_valid_frames']/max(s['total_frames'],1)*100:.1f}%")
        logger.info("=" * 60)

    def close(self):
        """释放资源"""
        self.detector.close()


def main():
    parser = argparse.ArgumentParser(
        description='从视频文件批量采集手语关键点数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/collect_from_video.py -i data/videos/
  python tools/collect_from_video.py -i data/videos/ -p K --frame-skip 2
  python tools/collect_from_video.py -i data/videos/ -o data/raw/collected --min-confidence 0.7
        """
    )
    parser.add_argument(
        '-i', '--input-dir', required=True,
        help='视频文件目录（支持 .mp4 .avi .mov .mkv .webm）'
    )
    parser.add_argument(
        '-o', '--output-dir', default=None,
        help='数据输出目录（默认: data/raw/collected）'
    )
    parser.add_argument(
        '-p', '--person-id', default=None,
        help='人员ID（文件名中未包含时使用此值）'
    )
    parser.add_argument(
        '--frame-skip', type=int, default=1,
        help='跳帧间隔：每N帧采样1帧（默认: 1，即全部帧）'
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.5,
        help='MediaPipe 检测最低置信度（默认: 0.5）'
    )
    parser.add_argument(
        '--vocab', type=str, default=None,
        help='词汇表路径（默认: data/vocab.csv）'
    )

    args = parser.parse_args()

    # 输出目录默认值
    output_dir = args.output_dir or str(config.raw_data_dir)

    logger.info("=" * 60)
    logger.info("聆心手语视频数据采集工具")
    logger.info("=" * 60)

    collector = VideoCollector(
        min_confidence=args.min_confidence,
        frame_skip=args.frame_skip,
        person_id=args.person_id,
        vocab_path=args.vocab,
    )

    try:
        collector.process_directory(args.input_dir, output_dir)
    except KeyboardInterrupt:
        logger.info("\n用户中断")
        collector._print_summary()
    finally:
        collector.close()


if __name__ == '__main__':
    main()
