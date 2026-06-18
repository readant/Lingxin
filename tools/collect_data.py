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
- 文件内容: numpy数组，形状为(n_frames, 171)
  - 171 = 21个手部关键点 × 3坐标 × 2只手 + 15个姿态关键点 × 3坐标
  - 0-62: 左手 (21×3=63)
  - 63-125: 右手 (21×3=63)
  - 126-170: 姿态 (15×3=45)
  - 坐标已归一化: x∈[0,1], y∈[0,1], z 保持原始值

使用示例：
>>> python tools/collect_data.py
请输入录制人ID: person001
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 必须在所有其他 import 之前设置环境变量
# 原因：cv2 / mediapipe 在加载时即初始化 C++ 运行时（glog / TFLite），
#       若 env var 设置晚于 C++ 静态初始化则完全无效。
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys

# 抑制 MediaPipe glog 输出到 stderr（同步 I/O 会阻塞主线程）
if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel'] = '3'       # 0=INFO 1=WARNING 2=ERROR 3=FATAL
if 'GLOG_logtostderr' not in os.environ:
    os.environ['GLOG_logtostderr'] = '0'        # 禁止写 stderr
if 'GLOG_alsologtostderr' not in os.environ:
    os.environ['GLOG_alsologtostderr'] = '0'    # 禁止 also-log-to-stderr
# 抑制 MediaPipe 警告信息（如 landmark_projection_calculator 警告）
if 'MEDIAPIPE_DISABLE_GPU' not in os.environ:
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # 保持GPU加速
# 禁用 clearcut 遥测上传（国内网络连 Google 超时 20-30s）
if 'MEDIAPIPE_DISABLE_ANALYTICS' not in os.environ:
    os.environ['MEDIAPIPE_DISABLE_ANALYTICS'] = '1'
# 抑制 TensorFlow Lite 日志（MediaPipe 底层依赖 TFLite）
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # 0=all 1=INFO 2=WARNING 3=ERROR

import cv2
import gc
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.hand_detector import HolisticDetector
from src.constants import HAND_CONNECTIONS, POSE_CONNECTIONS_HOLISTIC
from src.utils.logger import get_logger
import json
import time
import threading
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# 注意：此处不再将 stderr 重定向到 /dev/null
# 原因：会导致 input() 函数的提示信息无法显示
# MediaPipe 的日志已通过环境变量抑制（见上方 env var 设置）
# ═══════════════════════════════════════════════════════════════════════════════


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

    def __init__(self, person_id, save_dir='data/raw/collected', target_samples=30, vocab_path='data/vocab.csv'):
        """
        初始化数据采集器

        Args:
            person_id (str): 录制人员唯一标识
            save_dir (str, optional): 数据保存目录. Defaults to 'data/raw/collected'.
            target_samples (int, optional): 每个词汇的目标录制数量. Defaults to 30.
            vocab_path (str, optional): 词汇表路径. Defaults to 'data/vocab.csv'.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.detector = HolisticDetector()
        self.person_id = person_id
        self.save_dir = save_dir
        self.target_samples = target_samples

        # 从词汇表文件加载词汇列表
        self.vocab_df = pd.read_csv(vocab_path)
        self.words = self.vocab_df['word'].tolist()
        self.current_idx = 0  # 当前词汇索引

        # 预构建 word→category 映射字典，避免每帧 DataFrame 布尔过滤
        # （O(N) 每帧 → O(1) dict 查找）
        self.word_category = dict(
            zip(self.vocab_df['word'], self.vocab_df['category'])
        )

        # 录制状态管理
        self.is_recording = False           # 是否正在录制
        self.current_sequence = []           # 当前录制的关键点序列
        self.recorded_counts = {}           # 每个词汇已录制的数量
        self.frame_shape = None            # 当前帧尺寸（用于坐标归一化）

        # 异步保存状态（防止录制与保存竞争资源）
        self._is_saving = False             # 是否正在后台保存
        self._save_lock = threading.Lock()  # 保存锁

        # 性能监控（运行时显示帧率）
        self._perf_fps = 0.0               # 当前帧率
        self._perf_frame_times = []        # 最近帧时间列表（用于计算平均帧率）
        self._perf_last_time = time.time() # 上一帧时间

        # 初始化保存目录和已录制统计
        self._init_save_dirs()
        self._load_recorded_counts()

        # 初始化中文字体（用于UI显示）
        self._init_font()

        # 预渲染静态UI元素（只渲染一次，避免逐帧PIL开销）
        self._pre_render_static()

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

    def _pre_render_static(self):
        """
        预渲染所有可缓存的 UI 元素为 RGBA numpy 数组

        每帧只需 OpenCV alpha 合成，彻底消除 PIL→BGR 往返。
        """
        # 提示栏（静态，渲染一次）
        self._hint_rgba = None
        hint_text = "[SPACE]录制 [N]下一个 [P]上一个 [R]删除 [Q]统计退出 [ESC]直接退出"
        try:
            self._hint_rgba = self._render_text_rgba(
                hint_text, font_size=16, color=(200, 200, 200),
                bg_color=(0, 0, 0, 160))
        except Exception:
            pass

        # 动态文本缓存（只在内容变化时重新渲染）
        self._cached_word = None          # 当前词汇
        self._cached_word_rgba = None     # "词: 你好 (问候)" RGBA
        self._cached_progress_rgba = None # "进度: 3/50" RGBA
        self._cached_counts_rgba = None   # "已录: 5/30" RGBA
        self._cached_total_rgba = None    # "总计: 12" RGBA（仅当 total!=recorded 时显示）
        self._cached_rec_text = None      # 录制指示器文字
        self._cached_rec_rgba = None      # "RECORDING: 15 frames" RGBA
        self._cached_fps_text = None      # 帧率文字缓存
        self._cached_fps_rgba = None      # "FPS: 28" RGBA

    def _flush_capture(self, cap, max_frames=60):
        """
        丢弃摄像头缓冲区中的积压帧（增强版）

        在阻塞操作（_show_review / _show_countdown）之后调用，
        确保下一次 cap.read() 拿到的是实时帧而非积压旧帧。

        Args:
            cap: 摄像头捕获对象
            max_frames: 最大丢弃帧数（默认60帧，约2秒@30fps）
        """
        # 连续丢弃帧直到缓冲区为空
        # Windows下摄像头缓冲区通常有2-3秒的帧积压（60-90帧）
        for _ in range(max_frames):
            cap.grab()  # grab() 比 read() 更快，只获取不解码

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
        return self.recorded_counts.get(word, 0) + 1

    def _save_sequence(self, word, sequence):
        """
        保存关键点序列到文件

        对序列进行预处理：
        - 长度少于15帧的序列不予保存
        - 长度超过150帧的序列进行中心裁剪
        - x,y 坐标归一化到 [0,1]（除以帧宽高），消除分辨率依赖

        Args:
            word (str): 词汇名称
            sequence (list): 关键点序列（像素坐标）

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
        sequence_array = np.array(sequence, dtype=np.float32)

        # 坐标归一化：x,y 除以帧宽高 → [0,1] 范围，消除分辨率依赖
        if self.frame_shape is not None:
            h, w = self.frame_shape[:2]
            sequence_array[:, 0::3] /= w   # x 坐标归一化
            sequence_array[:, 1::3] /= h   # y 坐标归一化
            # z 坐标保持原始值

        # 生成文件名并保存
        index = self._get_next_index(word)
        file_name = f'{self.person_id}_{index:03d}.npy'
        save_path = os.path.join(self.save_dir, word, file_name)
        np.save(save_path, sequence_array)

        # 保存元信息
        metadata = {
            'person_id': self.person_id,
            'word': word,
            'category': self.word_category.get(word, ''),
            'num_frames': len(sequence_array),
            'feature_dim': sequence_array.shape[1] if sequence_array.ndim > 1 else 0,
            'timestamp': datetime.now().isoformat(),
            'file_name': file_name,
            'normalized': self.frame_shape is not None,
        }
        if self.frame_shape is not None:
            metadata['original_resolution'] = {
                'width': int(self.frame_shape[1]),
                'height': int(self.frame_shape[0]),
            }

        meta_path = save_path.replace('.npy', '_meta.json')
        try:
            with open(meta_path, 'w', encoding='utf-8') as f_meta:
                json.dump(metadata, f_meta, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 元信息保存失败不影响主流程

        # 更新统计信息
        self.recorded_counts[word] = self.recorded_counts.get(word, 0) + 1
        self._cached_word = None   # 强制下次渲染时刷新计数显示

        return True, f"已保存: {save_path} (形状: {sequence_array.shape})"

    def _save_sequence_async(self, word, sequence):
        """
        异步保存关键点序列到文件（后台线程执行，不阻塞主线程）

        Args:
            word (str): 词汇名称
            sequence (list): 关键点序列

        Returns:
            tuple: (是否成功, 消息)
        """
        def save_task():
            with self._save_lock:
                self._is_saving = True
                try:
                    success, msg = self._save_sequence(word, sequence)
                    self.logger.info(msg)
                finally:
                    self._is_saving = False
            # 更新UI缓存（在锁外执行）
            self._cached_word = None

        thread = threading.Thread(target=save_task)
        thread.daemon = True
        thread.start()
        return True, "正在保存..."

    def _delete_last_sequence(self, word):
        """
        删除最后录制的样本（同时删除对应的元信息文件）

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
            npy_path = os.path.join(word_dir, last_file)
            meta_path = npy_path.replace('.npy', '_meta.json')
            os.remove(npy_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            self.recorded_counts[word] = max(0, self.recorded_counts.get(word, 1) - 1)
            self._cached_word = None   # 强制下次渲染时刷新计数显示
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

    def _render_text_rgba(self, text, font_size=20, color=(255,255,255), bg_color=None):
        """
        用 PIL 将文本渲染为 RGBA numpy 数组（仅在内容变化时调用一次）

        Args:
            text: 要渲染的文本
            font_size: 字体大小
            color: 文本颜色 (R,G,B)
            bg_color: 背景色 (R,G,B,A)，None 表示透明背景

        Returns:
            np.ndarray: RGBA 图像 (H, W, 4), dtype=uint8
        """
        font = None
        if font_size != 20:
            # 为不同字号加载对应字体
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/arial.ttf",
            ]
            for fp in font_paths:
                if os.path.exists(fp):
                    try:
                        font = ImageFont.truetype(fp, font_size)
                        break
                    except Exception:
                        continue
        if font is None:
            font = self.font  # fallback to default (size 20)

        # 测量文字尺寸
        temp = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0] + 4
        th = bbox[3] - bbox[1] + 4

        # 渲染
        img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        if bg_color:
            draw.rectangle([(0, 0), (tw-1, th-1)], fill=bg_color)
        draw.text((2, 0), text, font=font, fill=tuple(list(color) + [255]))
        return np.array(img)

    @staticmethod
    def _paste_rgba(bgr_frame, rgba, x, y):
        """
        将 RGBA numpy 数组 alpha-合成到 BGR 帧上（纯 OpenCV，无 PIL 参与）

        Args:
            bgr_frame: BGR numpy 帧 (H, W, 3)
            rgba: RGBA numpy 数组 (H, W, 4)
            x, y: 粘贴位置
        """
        h, w = rgba.shape[:2]
        fy, fx = max(0, y), max(0, x)
        fh = min(h, bgr_frame.shape[0] - fy)
        fw = min(w, bgr_frame.shape[1] - fx)
        if fh <= 0 or fw <= 0:
            return
        rgba_crop = rgba[:fh, :fw]
        # 切片后数据不连续，cv2.cvtColor 要求 C-contiguous，必须 ascontiguousarray
        alpha = rgba_crop[:, :, 3:4].astype(np.float32) / 255.0  # (fh, fw, 1) 用于 broadcast
        rgb = np.ascontiguousarray(rgba_crop[:, :, :3])           # (fh, fw, 3) C-contiguous
        bgr_patch = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        roi = bgr_frame[fy:fy+fh, fx:fx+fw].astype(np.float32)
        blended = bgr_patch * alpha + roi * (1.0 - alpha)
        bgr_frame[fy:fy+fh, fx:fx+fw] = blended.astype(np.uint8)

    def _draw_text_pil(self, pil_img, text, position, font_size=20, color=(255, 255, 255)):
        """
        在PIL图像上绘制中文文本（仅用于非实时路径：倒计时/回顾界面）
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
            landmarks: 关键点数据，形状为(171,)

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

    def _playback_sequence(self, sequence, frame_h, frame_w):
        """
        回放录制的关键点序列

        以视频帧的形式逐帧显示序列中的关键点。
        自动检测归一化坐标并反归一化到像素空间。
        使用 RGBA 预渲染文本避免逐帧 PIL 开销。

        Args:
            sequence: 关键点序列（像素坐标或归一化坐标均可）
            frame_h (int): 显示帧高度
            frame_w (int): 显示帧宽度
        """
        h, w = frame_h, frame_w
        total_frames = len(sequence)

        self.logger.info(f"预览序列长度: {total_frames}")
        if total_frames > 0:
            self.logger.debug(f"第一帧关键点数据: {sequence[0][:10]}...")
            self.logger.debug(f"是否有手部数据: {not np.all(sequence[0][:126] == 0)}")

        # 自动检测归一化坐标：若全部 x,y 值 ∈ [0,1] 且 z 方向有非零值 → 反归一化
        needs_denorm = False
        if total_frames > 0:
            sample = np.asarray(sequence[0], dtype=np.float32)
            x_coords = sample[0::3]
            y_coords = sample[1::3]
            if np.max(x_coords) <= 1.0 and np.max(y_coords) <= 1.0:
                needs_denorm = True
                self.logger.info("检测到归一化坐标，自动反归一化到像素空间")

        # 预渲染静态文本为 RGBA numpy 数组，避免每帧 PIL 往返
        _hint_rgba = self._render_text_rgba(
            "[SPACE]暂停 [ESC]退出", font_size=18, color=(200, 200, 200))
        _no_data_rgba = self._render_text_rgba(
            "无有效关键点数据", font_size=24, color=(255, 0, 0))

        # title_text 是动态的（显示进度），每 5 帧重新渲染一次即可
        _cached_title_text = None
        _cached_title_rgba = None

        try:
            for idx, landmarks in enumerate(sequence):
                # 反归一化：将 [0,1] 范围坐标映射回像素坐标
                if needs_denorm:
                    landmarks = landmarks.copy()
                    landmarks[0::3] *= w   # x → 像素
                    landmarks[1::3] *= h   # y → 像素

                # 创建深灰色背景
                frame = np.ones((h, w, 3), dtype=np.uint8) * 40

                # 检查数据有效性
                has_hand = not np.all(landmarks[:126] == 0)
                has_pose = not np.all(landmarks[126:] == 0)

                if has_hand or has_pose:
                    frame = self._draw_landmarks_on_frame(frame, landmarks)
                else:
                    # 使用预渲染 RGBA 文本（无 PIL 参与）
                    self._paste_rgba(frame, _no_data_rgba,
                                     (w - _no_data_rgba.shape[1]) // 2, h // 2)

                # 添加边框
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)

                # 标题文字（每 5 帧或文字变化时重新渲染，减少文本渲染开销）
                title_text = f"预览回放  {idx + 1}/{total_frames} 帧"
                if _cached_title_text != title_text or idx % 5 == 0:
                    _cached_title_text = title_text
                    _cached_title_rgba = self._render_text_rgba(
                        title_text, font_size=24, color=(255, 255, 255))

                # 粘贴预渲染文本（纯 OpenCV alpha 合成，无 PIL）
                self._paste_rgba(frame, _cached_title_rgba, 10, 30)
                self._paste_rgba(frame, _hint_rgba, 10, h - 30)

                cv2.imshow('Preview', frame)

                key = cv2.waitKey(33) & 0xFF
                if key == 27:
                    return  # 通过 finally 块确保窗口清理
                elif key == ord(' '):
                    # 暂停等待下一次空格或ESC（使用短超时避免窗口失焦时卡死）
                    while True:
                        pause_key = cv2.waitKey(33) & 0xFF
                        if pause_key == ord(' ') or pause_key == 27 or pause_key == 255:
                            break

            cv2.waitKey(500)
        finally:
            # ═══════════════════════════════════════════════════════════════════
            # 关键：必须销毁 Preview 窗口，否则返回主循环后 OpenCV highgui
            # 需同时管理 Preview + Data Collection 两个窗口，
            # cv2.waitKey(1) 分摊处理所有窗口的事件 → 主循环帧率下降 → 卡顿
            # ═══════════════════════════════════════════════════════════════════
            cv2.destroyWindow('Preview')
            # 额外调用 waitKey 让 highgui 完成窗口销毁的底层清理
            for _ in range(3):
                cv2.waitKey(1)

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

            # 使用短超时循环等待，每帧清空非ESC按键（防止空格残留）
            wait_start = time.time()
            while (time.time() - wait_start) < 1.0:
                key = cv2.waitKey(33) & 0xFF
                if key == 27:  # ESC 取消
                    return False
                # 其他按键（包括空格）被忽略，不缓存
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
            key = cv2.waitKey(33) & 0xFF
            if key == ord(' '):
                return True, 'save'
            elif key in (ord('r'), ord('R')):
                return True, 'retry'
            elif key in (ord('d'), ord('D')):
                self._playback_sequence(sequence, h, w)
                # 预览回放期间摄像头持续捕获帧，缓冲区已积压
                # 必须清空，否则返回主循环后 cap.read() 拿到的是旧帧 → 感知延迟
                self._flush_capture(cap, max_frames=min(len(sequence) * 3, 200))
                cv2.imshow('Data Collection', frame)
            elif key == 27:
                return False, 'cancel'

    def _render_frame_ui(self, frame, status_text=""):
        """
        渲染所有 UI 元素到 BGR 帧（纯 OpenCV，零 PIL 参与）

        动态文本使用缓存 RGBA 数组，只在内容变化时重新渲染。
        每帧仅做几次 alpha 合成 + 边框绘制，毫秒级完成。

        Args:
            frame: 目标帧（BGR格式）
            status_text (str, optional): 状态文本. Defaults to "".

        Returns:
            numpy.ndarray: 绘制了所有UI的帧（BGR格式）
        """
        h, w = frame.shape[:2]

        # --- 词汇 + 类别信息（仅词汇切换时重新渲染） ---
        word = self.words[self.current_idx]
        if self._cached_word != word:
            self._cached_word = word
            category = self.word_category.get(word, '')
            self._cached_word_rgba = self._render_text_rgba(
                f"词: {word} ({category})", font_size=28, color=(255, 255, 255))
            self._cached_progress_rgba = self._render_text_rgba(
                f"进度: {self.current_idx + 1}/{len(self.words)}",
                font_size=22, color=(200, 200, 200))
            # 更新计数缓存（词变了，计数肯定也变了）
            recorded = self.recorded_counts.get(word, 0)
            total = self.total_counts.get(word, 0)
            self._cached_counts_rgba = self._render_text_rgba(
                f"已录: {recorded}/{self.target_samples}",
                font_size=22, color=(100, 255, 100))
            self._cached_total_rgba = (
                self._render_text_rgba(f"总计: {total}", font_size=16, color=(255, 255, 150))
                if total != recorded else None)

        # --- 帧率显示（仅当FPS变化超过2帧时重新渲染） ---
        fps_text = f"FPS: {int(self._perf_fps)}"
        if self._cached_fps_text != fps_text:
            self._cached_fps_text = fps_text
            # 根据帧率选择颜色：绿色(>=28)、黄色(20-27)、红色(<20)
            fps_color = (100, 255, 100) if self._perf_fps >= 28 else \
                        (255, 255, 100) if self._perf_fps >= 20 else (255, 100, 100)
            self._cached_fps_rgba = self._render_text_rgba(
                fps_text, font_size=18, color=fps_color)

        # --- 录制指示器（优化：仅当帧数变化超过5帧或状态切换时重新渲染） ---
        if self.is_recording:
            frame_count = len(self.current_sequence)
            # 只在帧数变化超过5帧时重渲染，避免每帧都渲染
            if self._cached_rec_text is None or abs(frame_count - int(self._cached_rec_text.split()[-2])) > 5:
                rec_text = f"RECORDING: {frame_count} frames"
                self._cached_rec_text = rec_text
                self._cached_rec_rgba = self._render_text_rgba(
                    rec_text, font_size=20, color=(255, 0, 0))
        else:
            self._cached_rec_text = None
            self._cached_rec_rgba = None

        # --- 粘贴预渲染元素（纯 OpenCV alpha 合成，无 PIL） ---
        # 左上：词名 + 进度
        if self._cached_word_rgba is not None:
            self._paste_rgba(frame, self._cached_word_rgba, 10, 10)
            self._paste_rgba(frame, self._cached_progress_rgba, 10, 45)

        # 右上：已录制数量 + 帧率
        if self._cached_counts_rgba is not None:
            ch = self._cached_counts_rgba.shape[0]
            self._paste_rgba(frame, self._cached_counts_rgba, w - 160, 10)
            if self._cached_total_rgba is not None:
                self._paste_rgba(frame, self._cached_total_rgba, w - 160, 10 + ch + 2)
                ch += self._cached_total_rgba.shape[0] + 2
            # 帧率显示在已录数量下方
            if self._cached_fps_rgba is not None:
                self._paste_rgba(frame, self._cached_fps_rgba, w - 100, 10 + ch + 2)

        # 左下：录制指示器
        if self._cached_rec_rgba is not None:
            rh = self._cached_rec_rgba.shape[0]
            self._paste_rgba(frame, self._cached_rec_rgba, 10, h - rh - 40)

        # 中央：状态消息（每帧内容不同，直接渲染不缓存）
        if status_text:
            status_rgba = self._render_text_rgba(status_text, font_size=26, color=(0, 255, 255))
            sw = status_rgba.shape[1]
            self._paste_rgba(frame, status_rgba, (w - sw) // 2, h // 2)

        # 底部：键盘提示栏（静态，渲染一次）
        if self._hint_rgba is not None:
            hh = self._hint_rgba.shape[0]
            self._paste_rgba(frame, self._hint_rgba, 10, h - hh - 4)

        # --- 状态边框（OpenCV 原生绘制） ---
        if self.is_recording:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
        else:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)

        return frame

    def run(self):
        """
        运行数据采集主循环

        持续从摄像头读取帧，检测关键点，并根据用户输入执行相应操作。
        按 ESC 键或 Q 键退出。
        """
        # Windows 下必须指定 CAP_DSHOW 后端，MSMF（默认）可能引发：
        # - 摄像头无法打开或间歇性黑屏
        # - 帧缓冲区积压导致延迟越来越大
        # - 与 MediaPipe 的线程模型冲突
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.logger.error("错误：无法打开摄像头")
            return

        try:
            # 降低分辨率减轻计算负载（640x480 足够关键点检测）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # 限制摄像头缓冲区大小，防止帧积压（Windows上可能不生效，由_flush_capture兜底）
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 预热：丢弃前几帧让自动曝光/白平衡稳定
            for _ in range(10):
                cap.grab()

            self.logger.info(f"开始数据采集，当前录制人: {self.person_id}")
            self.logger.info(f"词汇表共 {len(self.words)} 个词，每个词目标录制 {self.target_samples} 次")
            self.logger.info("操作说明：")
            self.logger.info("  [空格] 开始录制（3秒倒计时）")
            self.logger.info("  [N] 下一个词")
            self.logger.info("  [P] 上一个词")
            self.logger.info("  [R] 删除刚才录制的样本")
            self.logger.info("  [Q] 显示统计后退出")
            self.logger.info("  [ESC] 直接退出")

            MAX_RECORDING_FRAMES = 300  # 最大录制帧数（~10秒@30fps），防止误操作无限录制

            status_message = ""      # 状态消息文本
            status_until = 0.0       # 状态消息显示截止时间（time.time()秒数）
            frame_count = 0          # 帧计数器，用于周期性 GC
            recording_started_at = 0.0  # 录制开始时间戳（用于忽略启动瞬间的误按键）

            while True:
                # --- 帧率计算 ---
                current_time = time.time()
                frame_time = current_time - self._perf_last_time
                self._perf_last_time = current_time

                # 计算平均帧率（基于最近30帧）
                self._perf_frame_times.append(frame_time)
                if len(self._perf_frame_times) > 30:
                    self._perf_frame_times.pop(0)
                if self._perf_frame_times:
                    avg_frame_time = sum(self._perf_frame_times) / len(self._perf_frame_times)
                    self._perf_fps = 1.0 / avg_frame_time

                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # 水平翻转（镜像）
                self.frame_shape = frame.shape  # 记录帧尺寸，供 _save_sequence 归一化使用

                # BGR→RGB 仅用于 MediaPipe 推理（一次转换，共享给 hand + pose 两个模型）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.detect(frame, rgb_image=frame_rgb)
                landmarks = self.detector.get_landmarks(results, frame.shape)

                # UI 渲染（纯 OpenCV + 预渲染 RGBA 合成，无 PIL 往返）
                frame = self._render_frame_ui(
                    frame,
                    status_message if time.time() < status_until else "")
                # 关键点骨架（OpenCV 原生绘制）
                frame = self.detector.draw_landmarks(frame, results)

                # 检查是否检测到手部（阈值法，容忍微量噪声；any 短路退出）
                has_hand = bool(np.any(np.abs(landmarks[:126]) > 1e-4))

                # 录制状态处理
                if self.is_recording:
                    if has_hand:
                        self.current_sequence.append(landmarks)
                        # 达到最大录制帧数时自动停止
                        if len(self.current_sequence) >= MAX_RECORDING_FRAMES:
                            self.is_recording = False
                            self.logger.info(f"已达到最大录制帧数 {MAX_RECORDING_FRAMES}，自动停止")
                            # 先复制序列，防止后续清空影响后台保存
                            sequence_copy = self.current_sequence.copy()
                            proceed, choice = self._show_review(cap, sequence_copy)
                            self._flush_capture(cap)
                            gc.collect()  # 审查期间积压帧和PIL对象
                            if proceed and choice == 'save':
                                # 使用异步保存，不阻塞主线程
                                success, msg = self._save_sequence_async(
                                    self.words[self.current_idx], sequence_copy)
                                if success:
                                    status_message = msg
                                    status_until = time.time() + 1.0
                                else:
                                    status_message = msg
                                    status_until = time.time() + 3.0
                            elif choice == 'retry':
                                status_message = "已取消，请重新录制"
                                status_until = time.time() + 2.0
                            else:
                                status_message = "已取消录制"
                                status_until = time.time() + 2.0
                            self.current_sequence = []
                    else:
                        # 手部丢失时显示警告（仅在当前无消息时覆盖）
                        if time.time() >= status_until:
                            status_message = "警告：检测不到手！"
                            status_until = time.time() + 1.0

                # 周期性垃圾回收，防止长时间运行内存堆积
                frame_count += 1
                if frame_count % 300 == 0:
                    gc.collect()

                cv2.imshow('Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF

                # 按键处理
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # 空格：开始/停止录制
                    if self.is_recording:
                        # 忽略录制启动后 1 秒内的空格（防止倒计时残留按键误触发）
                        if time.time() - recording_started_at < 1.0:
                            continue
                        # 停止录制
                        self.is_recording = False
                        if len(self.current_sequence) > 0:
                            # 先复制序列，防止后续清空影响后台保存
                            sequence_copy = self.current_sequence.copy()
                            proceed, choice = self._show_review(cap, sequence_copy)
                            self._flush_capture(cap)  # 丢弃阻塞期间积压的旧帧
                            gc.collect()  # 审查期间积压帧和PIL对象
                            if proceed and choice == 'save':
                                # 使用异步保存，不阻塞主线程
                                success, msg = self._save_sequence_async(
                                    self.words[self.current_idx], sequence_copy)
                                if success:
                                    status_message = msg
                                    status_until = time.time() + 1.0
                                else:
                                    status_message = msg
                                    status_until = time.time() + 3.0
                            elif choice == 'retry':
                                status_message = "已取消，请重新录制"
                                status_until = time.time() + 2.0
                            else:
                                status_message = "已取消录制"
                                status_until = time.time() + 2.0
                            self.current_sequence = []
                        else:
                            status_message = "未录制到有效数据"
                            status_until = time.time() + 2.0
                    else:
                        # 开始录制
                        if not has_hand:
                            status_message = "请先伸出手！"
                            status_until = time.time() + 2.0
                            continue

                        # 等待后台保存完成（避免资源竞争）
                        if self._is_saving:
                            status_message = "请稍候，正在保存..."
                            status_until = time.time() + 0.5
                            continue

                        can_proceed = self._show_countdown(cap, seconds=3)
                        self._flush_capture(cap, max_frames=90)  # 丢弃倒计时期间积压的旧帧（增加到90帧）
                        gc.collect()  # 倒计时期间积累了大量临时对象
                        if not can_proceed:
                            status_message = "已取消录制"
                            status_until = time.time() + 2.0
                            continue

                        # 消费倒计时期间可能残留的按键（防止空格误触发停止）
                        # 多次调用 waitKey 彻底清空按键缓冲区
                        for _ in range(10):
                            key_flush = cv2.waitKey(10) & 0xFF
                            # 如果在清空过程中检测到ESC，取消录制
                            if key_flush == 27:
                                status_message = "已取消录制"
                                status_until = time.time() + 2.0
                                continue
                        self._flush_capture(cap, max_frames=10)

                        self.is_recording = True
                        self.current_sequence = []
                        recording_started_at = time.time()  # 记录启动时间
                        status_message = "录制中...按空格停止"
                        status_until = time.time() + 1.0
                elif key in (ord('n'), ord('N')):  # 下一个词汇
                    if self.current_idx < len(self.words) - 1:
                        self.current_idx += 1
                        status_message = f"切换到: {self.words[self.current_idx]}"
                        status_until = time.time() + 2.0
                elif key in (ord('p'), ord('P')):  # 上一个词汇
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        status_message = f"切换到: {self.words[self.current_idx]}"
                        status_until = time.time() + 2.0
                elif key in (ord('r'), ord('R')):  # 删除最后样本
                    success, msg = self._delete_last_sequence(self.words[self.current_idx])
                    self.logger.info(msg)
                    status_message = msg
                    status_until = time.time() + 2.0
                elif key in (ord('q'), ord('Q')):  # 退出
                    break
        finally:
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
            category = self.word_category.get(word, '')
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
    支持从环境变量 LINGXIN_PERSON_ID 和 LINGXIN_TARGET_SAMPLES 读取参数。
    """
    import argparse
    parser = argparse.ArgumentParser(description='聆心手语数据采集工具')
    parser.add_argument('--vocab', default='data/vocab.csv', help='词汇表路径 (默认: data/vocab.csv)')
    parser.add_argument('--output', default='data/raw/collected', help='数据保存目录 (默认: data/raw/collected)')
    parser.add_argument('--person-id', default=None, help='录制人ID (可选，也可通过环境变量 LINGXIN_PERSON_ID 设置)')
    parser.add_argument('--target-samples', type=int, default=None, help='每个词汇目标数量 (可选，也可通过环境变量 LINGXIN_TARGET_SAMPLES 设置)')
    args = parser.parse_args()

    logger = get_logger("DataCollector")
    logger.info("=" * 50)
    logger.info("聆心手语数据采集工具")
    logger.info("=" * 50)
    sys.stdout.flush()

    # 优先使用命令行参数，其次环境变量，最后交互输入
    person_id = args.person_id or os.environ.get('LINGXIN_PERSON_ID', '')
    target_samples = args.target_samples or int(os.environ.get('LINGXIN_TARGET_SAMPLES', '30'))

    if not person_id:
        person_id = input("请输入录制人ID: ").strip()

    if not person_id:
        logger.error("错误：录制人ID不能为空")
        return

    collector = DataCollector(
        person_id=person_id,
        save_dir=args.output,
        target_samples=target_samples,
        vocab_path=args.vocab
    )
    collector.run()


if __name__ == '__main__':
    main()
