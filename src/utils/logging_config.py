"""
Centralized logging configuration.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_to_console: Whether to log to console
        format_string: Custom format string
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log configuration
    root_logger.info("="*60)
    root_logger.info(f"Logging configured: level={level}, file={log_file}")
    root_logger.info("="*60)


def create_run_logger(
    run_name: str,
    log_dir: Path,
    level: str = "INFO"
) -> logging.Logger:
    """
    Create a logger for a specific run with timestamped file.
    
    Args:
        run_name: Name of the run (e.g., 'training', 'inference')
        log_dir: Directory for log files
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"
    
    # Setup logging
    setup_logging(level=level, log_file=log_file)
    
    # Return logger
    logger = logging.getLogger(run_name)
    logger.info(f"Run logger created: {run_name}")
    
    return logger


class LoggerMixin:
    """
    Mixin class to add logger to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.logger.info("Initialized")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for the class"""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)


def log_dataframe_info(df, name: str = "DataFrame") -> None:
    """
    Log useful information about a DataFrame.
    
    Args:
        df: DataFrame to log
        name: Name for the DataFrame
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"{name} info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'vehicle_id' in df.columns:
        logger.info(f"  Vehicles: {df['vehicle_id'].nunique()}")
    
    if 'timestamp' in df.columns:
        logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def log_section(title: str, width: int = 60) -> None:
    """
    Log a section separator.
    
    Args:
        title: Section title
        width: Width of separator line
    """
    logger = logging.getLogger(__name__)
    logger.info("")
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)