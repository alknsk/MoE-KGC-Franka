import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str = 'moe_kgc',
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 console: bool = True,
                 file: bool = True) -> logging.Logger:
    """
    Set up logger with console and file handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to add console handler
        file: Whether to add file handler

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f'Logging to {log_file}')

    return logger

def get_logger(name: str = 'moe_kgc') -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

class TensorBoardLogger:
    """TensorBoard logger wrapper"""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: Optional[int] = None):
        """Log multiple scalar values"""
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log histogram"""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)

    def log_graph(self, model, input_sample):
        """Log model graph"""
        self.writer.add_graph(model, input_sample)

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text"""
        if step is None:
            step = self.step
        self.writer.add_text(tag, text, step)

    def log_image(self, tag: str, image, step: Optional[int] = None):
        """Log image"""
        if step is None:
            step = self.step
        self.writer.add_image(tag, image, step)

    def increment_step(self):
        """Increment global step"""
        self.step += 1

    def close(self):
        """Close writer"""
        self.writer.close()