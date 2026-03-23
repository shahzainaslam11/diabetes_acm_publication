import logging
import os
from datetime import datetime


def get_logger(name: str = "experiment", log_dir: str = "experiments/logs"):
    """
    Creates a standardized logger for experiments.

    Design Philosophy (aligned with notebook):
    - Simple, readable outputs (like print statements)
    - Adds reproducibility via log files
    - Avoids excessive verbosity or external dependencies
    """

    # creating log directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    # timestamped log file (ensures experiment tracking)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)

    # prevented duplicate handlers (important in notebooks/scripts reruns)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # ===== Console Handler (Notebook-like behavior) =====
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_format)

    # ===== File Handler (Reproducibility) =====
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_format)

    # adding handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Logs will be saved to: {log_file}")

    return logger
