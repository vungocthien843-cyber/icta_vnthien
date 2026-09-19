import logging
import os
import sys


def setup_logger(name: str = "DDI", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Khởi tạo logger ghi ra console và file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Tránh gắn trùng lặp handlers nếu hàm được gọi nhiều lần
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Đảm bảo console trên Windows ghi được ký tự Tiếng Việt UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
