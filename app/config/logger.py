import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os

load_dotenv()

log_level = os.getenv('LOG_LEVEL', logging.INFO)
try:
    LOG_LEVEL = int(log_level)
except ValueError:
    # 若转换失败，使用默认日志级别
    LOG_LEVEL = logging.INFO

# from app.config.logger import setup_logger
# logger = setup_logger(__name__)

# 用于存储单例日志实例
_logger_instance = None

def setup_logger(name: str | None = None,
                 log_file: str = "app.log",
                 max_size: int = 1024 * 1024 * 50,  # 单个日志文件最大为 50MB
                 backup_count: int = 5) -> logging.Logger:  # 保留 5 个旧日志文件
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = logging.getLogger(name)
        if not _logger_instance.handlers:
            _logger_instance.setLevel(LOG_LEVEL)

            # 创建文件处理器，支持日志文件轮转
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)

            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)

            # 添加处理器到日志器
            _logger_instance.addHandler(file_handler)
            _logger_instance.addHandler(console_handler)
    return _logger_instance