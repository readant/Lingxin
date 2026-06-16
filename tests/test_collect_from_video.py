"""测试 tools/collect_from_video.py — 视频文件名解析"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.collect_from_video import parse_filename


class TestParseFilename:
    """测试视频文件名解析 — 确保各种命名格式都能正确提取信息"""

    @pytest.mark.parametrize("filename,expected_word,expected_person,expected_index", [
        # 标准三段式
        ("你好_A_001.mp4", "你好", "A", 1),
        ("谢谢_B_010.mp4", "谢谢", "B", 10),
        # 两段式（无序号）
        ("再见_C.mp4", "再见", "C", None),
        ("你好_mike.mp4", "你好", "mike", None),
        # 单段式（纯词汇）
        ("帮助.mp4", "帮助", None, None),
        # 数字人员ID
        ("你好_001_002.mp4", "你好", "001", 2),
        # 多字符人员ID
        ("谢谢_user123.mp4", "谢谢", "user123", None),
    ])
    def test_filename_formats(self, filename, expected_word, expected_person, expected_index):
        """各种命名格式应正确解析"""
        result = parse_filename(filename)
        assert result['word'] == expected_word
        assert result['person_id'] == expected_person
        assert result['index'] == expected_index

    @pytest.mark.parametrize("ext", ["mp4", "avi", "mov", "mkv", "webm"])
    def test_video_extensions(self, ext):
        """应支持所有视频格式"""
        result = parse_filename(f"你好_A.{ext}")
        assert result['ext'] == ext

    def test_case_insensitive_extension(self):
        """扩展名大小写不敏感"""
        assert parse_filename("你好_A.MP4")['ext'] == 'MP4'
        assert parse_filename("你好_A.Mp4")['ext'] == 'Mp4'

    def test_full_path_extracts_filename(self):
        """完整路径应正确提取文件名"""
        result = parse_filename("data/videos/你好_A_001.mp4")
        assert result['word'] == '你好'
        assert result['person_id'] == 'A'
        assert result['index'] == 1

    def test_windows_path(self):
        """Windows 风格路径"""
        result = parse_filename("data\\videos\\谢谢_B.mp4")
        assert result['word'] == '谢谢'
        assert result['person_id'] == 'B'

    def test_no_extension_falls_back(self):
        """无扩展名文件应作为纯词汇名"""
        result = parse_filename("你好_A")
        assert result['word'] == '你好_A'
        assert result['person_id'] is None
