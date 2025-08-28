import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

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
                 log_file: str = None,
                 max_size: int = 1024 * 1024 * 50,  # 单个日志文件最大为 50MB
                 backup_count: int = 5) -> logging.Logger:  # 保留 5 个旧日志文件
    global _logger_instance

    # 如果没有指定日志文件，则使用默认路径
    if log_file is None:
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        # 创建统一的日志目录
        log_dir = project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # 设置默认日志文件路径
        log_file = str(log_dir / "app.log")
    
    if _logger_instance is None:
        _logger_instance = logging.getLogger(name)
        if not _logger_instance.handlers:
            _logger_instance.setLevel(LOG_LEVEL)

            # 创建文件处理器，支持日志文件轮转
            log_path = Path(log_file)
            log_dir = log_path.parent
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