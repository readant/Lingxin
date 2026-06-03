"""
DataCollector - 手语数据采集工具

本文件实现了实时手语数据采集功能，通过摄像头捕获手部关键点数据。

工作流程：
1. 从摄像头读取视频帧
2. 使用 MediaPipe Holistic 检测手部、姿态关键点
3. 用户通过键盘控制录制流程
4. 将录制的关键点序列保存为 .npy 文件

键盘控制：
- [空格]: 开始/停止录制
- [N]: 下一个词汇
- [P]: 上一个词汇
- [R]: 删除最后录制的样本
- [Q]: 显示统计后退出
- [ESC]: 直接退出

数据保存格式：
- 文件名: {person_id}_{序号:03d}.npy
- 文件内容: numpy数组，形状为(n_frames, 153)
  - 153 = 21个手部关键点 × 3坐标 × 2只手 + 15个姿态关键点 × 3坐标
  - 前63个: 左手 (21×3)
  - 63-125个: 右手 (21×3)
  - 126-152个: 姿态 (15×3)

使用示例：
>>> python tools/collect_data.py
请输入录制人ID: person001
"""

import cv2
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.hand_detector import HolisticDetector
from src.constants import HAND_CONNECTIONS, POSE_CONNECTIONS_HOLISTIC
from src.utils.logger import get_logger
import json
from datetime import datetime


class DataCollector:
    """
    手语数据采集器

    负责管理数据采集的完整流程，包括：
    - 视频捕获和关键点检测
    - 录制状态管理
    - 样本保存和删除
    - 用户界面渲染

    Attributes:
        detector: MediaPipe Holistic 检测器实例
        person_id: 录制人员ID
        save_dir: 数据保存根目录
        target_samples: 每个词汇的目标录制数量
        words: 词汇表列表
        current_idx: 当前正在录制的词汇索引
        is_recording: 是否正在录制
        current_sequence: 当前录制的关键点序列
        recorded_counts: 每个词汇已录制的样本数量
    """

    def __init__(self, person_id, save_dir='data/raw/collected', target_samples=30):
        """
        初始化数据采集器

        Args:
            person_id (str): 录制人员唯一标识
            save_dir (str, optional): 数据保存目录. Defaults to 'data/raw/collected'.
            target_samples (int, optional): 每个词汇的目标录制数量. Defaults to 30.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.detector = HolisticDetector()
        self.person_id = person_id
        self.save_dir = save_dir
        self.target_samples = target_samples

        # 从词汇表文件加载词汇列表
        vocab_path = 'data/vocab.csv'
        self.vocab_df = pd.read_csv(vocab_path)
        self.words = self.vocab_df['word'].tolist()
        self.current_idx = 0  # 当前词汇索引

        # 录制状态管理
        self.is_recording = False           # 是否正在录制
        self.current_sequence = []           # 当前录制的关键点序列
        self.recorded_counts = {}           # 每个词汇已录制的数量
        self.last_saved_sequence = None     # 最后保存的序列（用于回放）

        # 初始化保存目录和已录制统计
        self._init_save_dirs()
        self._load_recorded_counts()

        # 初始化中文字体（用于UI显示）
        self._init_font()

    def _init_font(self):
        """
        初始化中文字体

        按优先级尝试加载系统字体，确保UI可以正确显示中文。
        支持的字体：微软雅黑、黑体、宋体、 Arial
        """
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf", # 黑体
            "C:/Windows/Fonts/simsun.ttc", # 宋体
            "C:/Windows/Fonts/arial.ttf",  # Arial
        ]
        self.font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    self.font = ImageFont.truetype(fp, 20)
                    break
                except:
                    continue

        # 如果所有字体都加载失败，使用默认字体
        if self.font is None:
            self.font = ImageFont.load_default()

    def _init_save_dirs(self):
        """
        初始化数据保存目录

        为词汇表中的每个词汇创建对应的保存目录。
        目录结构: save_dir/word/
        """
        for word in self.words:
            word_dir = os.path.join(self.save_dir, word)
            os.makedirs(word_dir, exist_ok=True)

    def _load_recorded_counts(self):
        """
        加载已录制样本统计

        扫描保存目录，统计每个词汇的样本数量。
        同时记录当前人员的数量和所有人员的总量。
        用于显示录制进度。
        """
        # 初始化两个计数器
        self.recorded_counts = {}       # 当前人员的数量
        self.total_counts = {}          # 所有人员的总量

        for word in self.words:
            word_dir = os.path.join(self.save_dir, word)
            if os.path.exists(word_dir):
                # 统计当前录制人
                mine = [f for f in os.listdir(word_dir)
                       if f.startswith(self.person_id) and f.endswith('.npy')]
                self.recorded_counts[word] = len(mine)

                # 统计所有人员
                all_files = [f for f in os.listdir(word_dir)
                           if f.endswith('.npy')]
                self.total_counts[word] = len(all_files)
            else:
                self.recorded_counts[word] = 0
                self.total_counts[word] = 0

    def _get_next_index(self, word):
        """
        获取下一个样本的序号

        Args:
            word (str): 词汇名称

        Returns:
            int: 下一个可用的样本序号
        """
        word_dir = os.path.join(self.save_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(word_dir)
                        if f.startswith(self.person_id) and f.endswith('.npy')]
        return len(existing_files) + 1

    def _save_sequence(self, word, sequence):
        """
        保存关键点序列到文件

        对序列进行预处理：
        - 长度少于15帧的序列不予保存
        - 长度超过150帧的序列进行中心裁剪

        Args:
            word (str): 词汇名称
            sequence (list): 关键点序列

        Returns:
            tuple: (是否成功, 消息)
        """
        # 检查序列长度是否满足最低要求
        if len(sequence) < 15:
            return False, f"序列太短（{len(sequence)}帧），至少需要15帧"

        # 对过长序列进行中心裁剪
        if len(sequence) > 150:
            start = (len(sequence) - 150) // 2
            sequence = sequence[start:start + 150]

        # 转换为numpy数组
        sequence_array = np.array(sequence)

        # 生成文件名并保存
        index = self._get_next_index(word)
        file_name = f'{self.person_id}_{index:03d}.npy'
        save_path = os.path.join(self.save_dir, word, file_name)
        np.save(save_path, sequence_array)

        # 保存元信息
        metadata = {
            'person_id': self.person_id,
            'word': word,
            'category': self.vocab_df[self.vocab_df['word'] == word]['category'].values[0]
                if word in self.vocab_df['word'].values else '',
            'num_frames': len(sequence_array),
            'feature_dim': sequence_array.shape[1] if sequence_array.ndim > 1 else 0,
            'timestamp': datetime.now().isoformat(),
            'file_name': file_name,
        }
        meta_path = save_path.replace('.npy', '_meta.json')
        try:
            with open(meta_path, 'w', encoding='utf-8') as f_meta:
                json.dump(metadata, f_meta, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 元信息保存失败不影响主流程

        # 更新统计信息
        self.recorded_counts[word] = self.recorded_counts.get(word, 0) + 1
        self.last_saved_sequence = sequence_array.copy()

        return True, f"已保存: {save_path} (形状: {sequence_array.shape})"

    def _delete_last_sequence(self, word):
        """
        删除最后录制的样本

        Args:
            word (str): 词汇名称

        Returns:
            tuple: (是否成功, 消息)
        """
        word_dir = os.path.join(self.save_dir, word)
        existing_files = sorted([f for f in os.listdir(word_dir)
                                 if f.startswith(self.person_id) and f.endswith('.npy')])
        if existing_files:
            last_file = existing_files[-1]
            os.remove(os.path.join(word_dir, last_file))
            self.recorded_counts[word] = max(0, self.recorded_counts.get(word, 1) - 1)
            self.last_saved_sequence = None
            return True, f"已删除: {last_file}"
        return False, "没有可删除的录像"

    def _cv2_to_pil(self, frame):
        """
        将OpenCV图像格式转换为PIL图像格式

        Args:
            frame: OpenCV图像 (BGR格式)

        Returns:
            PIL.Image: PIL图像格式 (RGB模式)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _pil_to_cv2(self, pil_image):
        """
        将PIL图像格式转换回OpenCV图像格式

        Args:
            pil_image: PIL.Image图像 (RGB模式)

        Returns:
            numpy.ndarray: OpenCV图像 (BGR格式)
        """
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _draw_text_pil(self, pil_img, text, position, font_size=20, color=(255, 255, 255)):
        """
        在PIL图像上绘制中文文本

        Args:
            pil_img: PIL.Image图像
            text (str): 要绘制的文本
            position (tuple): 文本位置 (x, y)
            font_size (int, optional): 字体大小. Defaults to 20.
            color (tuple, optional): 文本颜色 (R, G, B). Defaults to (255, 255, 255).
        """
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=self.font, fill=color)

    def _create_blank_frame(self, h, w, color=(20, 20, 20)):
        """
        创建空白帧

        Args:
            h (int): 帧高度
            w (int): 帧宽度
            color (tuple, optional): 背景颜色 (B, G, R). Defaults to (20, 20, 20).

        Returns:
            numpy.ndarray: 空白帧图像
        """
        return np.full((h, w, 3), color, dtype=np.uint8)

    def _draw_landmarks_on_frame(self, frame, landmarks):
        """
        在帧上绘制关键点骨架

        Args:
            frame: 目标图像
            landmarks: 关键点数据，形状为(153,)

        Returns:
            numpy.ndarray: 绘制了关键点的图像
        """
        h, w = frame.shape[:2]

        # 解析关键点数据
        left_hand = landmarks[:63].reshape(21, 3)    # 左手21个点
        right_hand = landmarks[63:126].reshape(21, 3)  # 右手21个点
        pose = landmarks[126:].reshape(15, 3)        # 姿态15个点

        # 手部骨架连接定义（从共享常量导入）
        hand_connections = HAND_CONNECTIONS

        # 绘制左手（蓝色）
        for start, end in hand_connections:
            x1, y1 = int(left_hand[start][0]), int(left_hand[start][1])
            x2, y2 = int(left_hand[end][0]), int(left_hand[end][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for i in range(21):
            x, y = int(left_hand[i][0]), int(left_hand[i][1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # 绘制右手（绿色）
        for start, end in hand_connections:
            x1, y1 = int(right_hand[start][0]), int(right_hand[start][1])
            x2, y2 = int(right_hand[end][0]), int(right_hand[end][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i in range(21):
            x, y = int(right_hand[i][0]), int(right_hand[i][1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # 绘制姿态骨架（从共享常量导入）
        pose_connections = POSE_CONNECTIONS_HOLISTIC
        for start, end in pose_connections:
            x1, y1 = int(pose[start][0]), int(pose[start][1])
            x2, y2 = int(pose[end][0]), int(pose[end][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for i in range(15):
            x, y = int(pose[i][0]), int(pose[i][1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

        return frame

    def _playback_sequence(self, sequence, cap):
        """
        回放录制的关键点序列

        以视频帧的形式逐帧显示序列中的关键点。

        Args:
            sequence: 关键点序列
            cap: 摄像头捕获对象（用于获取帧尺寸）
        """
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.logger.info(f"预览序列长度: {len(sequence)}")
        if len(sequence) > 0:
            self.logger.debug(f"第一帧关键点数据: {sequence[0][:10]}...")
            self.logger.debug(f"是否有手部数据: {not np.all(sequence[0][:126] == 0)}")

        for idx, landmarks in enumerate(sequence):
            # 创建深灰色背景
            frame = np.ones((h, w, 3), dtype=np.uint8) * 40

            # 检查数据有效性
            has_hand = not np.all(landmarks[:126] == 0)
            has_pose = not np.all(landmarks[126:] == 0)

            if has_hand or has_pose:
                frame = self._draw_landmarks_on_frame(frame, landmarks)
            else:
                pil_frame = self._cv2_to_pil(frame)
                self._draw_text_pil(pil_frame, "无有效关键点数据",
                                    (w//2-100, h//2), font_size=24, color=(255, 0, 0))
                frame = self._pil_to_cv2(pil_frame)

            # 添加边框
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)

            # 添加文字信息
            pil_frame = self._cv2_to_pil(frame)
            self._draw_text_pil(pil_frame, f"预览回放  {idx + 1}/{len(sequence)} 帧",
                               (10, 30), font_size=24, color=(255, 255, 255))
            self._draw_text_pil(pil_frame, "[SPACE]暂停 [ESC]退出",
                               (10, h - 30), font_size=18, color=(200, 200, 200))

            frame = self._pil_to_cv2(pil_frame)
            cv2.imshow('Preview', frame)

            key = cv2.waitKey(33) & 0xFF
            if key == 27:
                return
            elif key == ord(' '):
                cv2.waitKey(0)

        cv2.waitKey(500)

    def _show_countdown(self, cap, seconds=3):
        """
        显示录制倒计时

        Args:
            cap: 摄像头捕获对象
            seconds (int, optional): 倒计时秒数. Defaults to 3.

        Returns:
            bool: 是否完成倒计时（False表示用户取消）
        """
        for i in range(seconds, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return False

            frame = cv2.flip(frame, 1)
            results = self.detector.detect(frame)
            frame = self.detector.draw_landmarks(frame, results)

            pil_frame = self._cv2_to_pil(frame)
            h, w = frame.shape[:2]

            # 绘制倒计时数字
            countdown_text = f"{i}"
            text_x = w // 2 - 50
            text_y = h // 2 - 50
            self._draw_text_pil(pil_frame, countdown_text,
                               (text_x, text_y), font_size=120, color=(0, 255, 255))

            self._draw_text_pil(pil_frame, "准备录制...",
                               (w // 2 - 80, h // 2 + 80), font_size=30, color=(255, 255, 255))
            self._draw_text_pil(pil_frame, "按 [ESC] 取消",
                               (w // 2 - 80, h // 2 + 120), font_size=20, color=(200, 200, 200))

            frame = self._pil_to_cv2(pil_frame)
            # 绘制黄色边框（在PIL转换后进行，确保可见）
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), 4)
            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1000) & 0xFF
            if key == 27:
                return False
        return True

    def _show_review(self, cap, sequence):
        """
        显示录制回顾界面

        录制完成后显示选项：保存、重录、预览或取消。

        Args:
            cap: 摄像头捕获对象
            sequence: 录制序列

        Returns:
            tuple: (是否继续, 用户选择)
        """
        ret, frame = cap.read()
        if not ret:
            return False, None

        frame = cv2.flip(frame, 1)
        results = self.detector.detect(frame)
        frame = self.detector.draw_landmarks(frame, results)

        pil_frame = self._cv2_to_pil(frame)
        h, w = frame.shape[:2]

        self._draw_text_pil(pil_frame, f"录制完成，共 {len(sequence)} 帧",
                           (w // 2 - 150, h // 2 - 60), font_size=28, color=(255, 255, 0))
        self._draw_text_pil(pil_frame, "[SPACE] 保存  [R] 重录  [D] 预览回放  [ESC] 取消",
                           (w // 2 - 220, h // 2 + 10), font_size=22, color=(255, 255, 255))

        frame = self._pil_to_cv2(pil_frame)
        # 绘制黄色边框（在PIL转换后进行，确保可见）
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 0), 4)
        cv2.imshow('Data Collection', frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):
                return True, 'save'
            elif key in (ord('r'), ord('R')):
                return True, 'retry'
            elif key in (ord('d'), ord('D')):
                self._playback_sequence(sequence, cap)
                cv2.imshow('Data Collection', frame)
            elif key == 27:
                return False, 'cancel'

    def _draw_ui(self, frame, status_text=""):
        """
        绘制主界面UI

        在帧上叠加显示词汇信息、录制状态、进度等。

        Args:
            frame: 目标帧
            status_text (str, optional): 状态文本. Defaults to "".

        Returns:
            numpy.ndarray: 绘制了UI的帧
        """
        pil_frame = self._cv2_to_pil(frame)
        h, w = frame.shape[:2]

        # 绘制文字（全部在PIL上完成，支持中文显示）
        if self.is_recording:
            rec_text = f"RECORDING: {len(self.current_sequence)} frames"
            self._draw_text_pil(pil_frame, rec_text, (10, h - 35),
                               font_size=20, color=(255, 0, 0))

        # 绘制词汇和类别信息
        word = self.words[self.current_idx]
        category = self.vocab_df.iloc[self.current_idx]['category']
        self._draw_text_pil(pil_frame, f"词: {word} ({category})",
                           (10, 10), font_size=28, color=(255, 255, 255))
        self._draw_text_pil(pil_frame, f"进度: {self.current_idx + 1}/{len(self.words)}",
                           (10, 45), font_size=22, color=(200, 200, 200))

        # 绘制已录制数量（个人 / 总计）
        recorded = self.recorded_counts.get(word, 0)
        total = self.total_counts.get(word, 0)
        self._draw_text_pil(pil_frame, f"已录: {recorded}/{self.target_samples}",
                           (w - 160, 10), font_size=22, color=(100, 255, 100))
        if total != recorded:
            self._draw_text_pil(pil_frame, f"总计: {total}",
                               (w - 160, 38), font_size=16, color=(255, 255, 150))

        # 绘制状态文本
        if status_text:
            self._draw_text_pil(pil_frame, status_text, (w // 2 - 150, h // 2),
                               font_size=26, color=(0, 255, 255))

        # 绘制操作提示
        self._draw_text_pil(pil_frame,
                           "[SPACE]录制 [N]下一个 [P]上一个 [R]删除 [Q]统计退出 [ESC]直接退出",
                           (10, h - 30), font_size=16, color=(150, 150, 150))

        # 转换回cv2，然后绘制状态边框（在PIL转换后进行，确保可见）
        frame = self._pil_to_cv2(pil_frame)
        if self.is_recording:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
        else:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

        return frame

    def _draw_warning(self, frame, message):
        """
        在帧上显示警告信息

        Args:
            frame: 目标帧
            message (str): 警告信息

        Returns:
            numpy.ndarray: 绘制了警告的帧
        """
        pil_frame = self._cv2_to_pil(frame)
        h, w = frame.shape[:2]
        self._draw_text_pil(pil_frame, message, (w // 2 - 150, h // 2 + 30),
                           font_size=26, color=(0, 255, 255))
        return self._pil_to_cv2(pil_frame)

    def run(self):
        """
        运行数据采集主循环

        持续从摄像头读取帧，检测关键点，并根据用户输入执行相应操作。
        按 ESC 键或 Q 键退出。
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("错误：无法打开摄像头")
            return

        self.logger.info(f"开始数据采集，当前录制人: {self.person_id}")
        self.logger.info(f"词汇表共 {len(self.words)} 个词，每个词目标录制 {self.target_samples} 次")
        self.logger.info("操作说明：")
        self.logger.info("  [空格] 开始录制（3秒倒计时）")
        self.logger.info("  [N] 下一个词")
        self.logger.info("  [P] 上一个词")
        self.logger.info("  [R] 删除刚才录制的样本")
        self.logger.info("  [Q] 显示统计后退出")
        self.logger.info("  [ESC] 直接退出")

        status_message = ""   # 状态消息
        status_timer = 0      # 状态消息显示时间

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 水平翻转（镜像）
            results = self.detector.detect(frame)
            landmarks = self.detector.get_landmarks(results, frame.shape)
            frame = self.detector.draw_landmarks(frame, results)

            # 检查是否检测到手部
            has_hand = not np.all(landmarks[:126] == 0)

            # 录制状态处理
            if self.is_recording:
                if has_hand:
                    self.current_sequence.append(landmarks)
                else:
                    # 手部丢失时显示警告
                    if status_timer <= 0:
                        status_message = "警告：检测不到手！"
                        status_timer = 30

            # 更新状态消息计时器
            if status_timer > 0:
                frame = self._draw_warning(frame, status_message)
                status_timer -= 1

            frame = self._draw_ui(frame, status_message if status_timer > 0 else "")

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF

            # 按键处理
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # 空格：开始/停止录制
                if self.is_recording:
                    # 停止录制
                    self.is_recording = False
                    if len(self.current_sequence) > 0:
                        proceed, choice = self._show_review(cap, self.current_sequence)
                        if proceed and choice == 'save':
                            success, msg = self._save_sequence(
                                self.words[self.current_idx], self.current_sequence)
                            self.logger.info(msg)
                            if success:
                                status_message = f"保存成功！({len(self.current_sequence)}帧)"
                                status_timer = 60
                            else:
                                status_message = msg
                                status_timer = 90
                        elif choice == 'retry':
                            status_message = "已取消，请重新录制"
                            status_timer = 60
                        else:
                            status_message = "已取消录制"
                            status_timer = 60
                        self.current_sequence = []
                    else:
                        status_message = "未录制到有效数据"
                        status_timer = 60
                else:
                    # 开始录制
                    if not has_hand:
                        status_message = "请先伸出手！"
                        status_timer = 60
                        continue

                    can_proceed = self._show_countdown(cap, seconds=3)
                    if not can_proceed:
                        status_message = "已取消录制"
                        status_timer = 60
                        continue

                    self.is_recording = True
                    self.current_sequence = []
                    status_message = "录制中...按空格停止"
                    status_timer = 30
            elif key in (ord('n'), ord('N')):  # 下一个词汇
                if self.current_idx < len(self.words) - 1:
                    self.current_idx += 1
                    status_message = f"切换到: {self.words[self.current_idx]}"
                    status_timer = 60
            elif key in (ord('p'), ord('P')):  # 上一个词汇
                if self.current_idx > 0:
                    self.current_idx -= 1
                    status_message = f"切换到: {self.words[self.current_idx]}"
                    status_timer = 60
            elif key in (ord('r'), ord('R')):  # 删除最后样本
                success, msg = self._delete_last_sequence(self.words[self.current_idx])
                self.logger.info(msg)
                status_message = msg
                status_timer = 60
            elif key in (ord('q'), ord('Q')):  # 退出
                break

        cap.release()
        cv2.destroyAllWindows()

        self._print_statistics()

        self.detector.close()

    def _print_statistics(self):
        """
        打印录制统计信息

        显示每个词汇的录制进度和总体完成率。
        """
        self.logger.info("\n" + "=" * 50)
        self.logger.info("录制统计")
        self.logger.info("=" * 50)

        total_mine = 0
        total_all = 0
        total_target = len(self.words) * self.target_samples

        for word in self.words:
            mine = self.recorded_counts.get(word, 0)
            all_count = self.total_counts.get(word, 0)
            total_mine += mine
            total_all += all_count
            category = self.vocab_df[self.vocab_df['word'] == word]['category'].values[0]
            status = "V" if mine >= self.target_samples else "X"
            if all_count != mine:
                self.logger.info(f"  {status} {word} ({category}): 个人{mine}/{self.target_samples} | 总计{all_count}")
            else:
                self.logger.info(f"  {status} {word} ({category}): {mine}/{self.target_samples}")

        self.logger.info("-" * 50)
        self.logger.info(f"个人: {total_mine}/{total_target} ({total_mine/total_target*100:.1f}%)")
        self.logger.info(f"总计: {total_all} (含所有人员)")
        self.logger.info("=" * 50)


def main():
    """
    主入口函数

    提示用户输入录制人ID，然后启动数据采集器。
    """
    logger = get_logger("DataCollector")
    logger.info("=" * 50)
    logger.info("聆心手语数据采集工具")
    logger.info("=" * 50)

    person_id = input("请输入录制人ID: ").strip()
    if not person_id:
        logger.error("错误：录制人ID不能为空")
        return

    collector = DataCollector(person_id=person_id)
    collector.run()


if __name__ == '__main__':
    main()
