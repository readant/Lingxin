"""
统一日志配置

为项目提供统一的 logging 配置，替代散落的 print() 调用。
支持控制台和文件输出，可按需配置级别和格式。

使用示例：
>>> from src.utils.logger import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("模型训练开始")
>>> logger.warning("数据量不足")
>>> logger.error("模型加载失败")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = 'lingxin',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回指定名称的 logger

    Args:
        name: logger 名称
        level: 日志级别，默认 INFO
        log_file: 日志文件路径，None 表示只输出到控制台
        log_format: 日志格式字符串

    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    if log_format is None:
        log_format = '[%(asctime)s] %(levelname)-7s %(name)s | %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')

        # 控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件 handler（可选）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取一个已配置的 logger。
    避免重复配置：如果 logger 已有 handler 则直接返回。

    Args:
        name: logger 名称，通常传入 __name__
        level: 日志级别

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-7s %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
