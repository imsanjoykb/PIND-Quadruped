import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    use_colors: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        use_colors: Whether to use colored output
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors and sys.stdout.isatty():
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save metrics
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.metrics = []
    
    def log(self, epoch: int, metrics: dict):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics.append(entry)
        
        # Append to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_metrics(self):
        """Get all logged metrics."""
        return self.metrics
    
    def get_best(self, metric: str, mode: str = 'min'):
        """
        Get best value for a metric.
        
        Args:
            metric: Metric name
            mode: 'min' or 'max'
        
        Returns:
            Best value and epoch
        """
        if not self.metrics:
            return None, None
        
        values = [(m['epoch'], m[metric]) for m in self.metrics if metric in m]
        
        if not values:
            return None, None
        
        if mode == 'min':
            best_epoch, best_value = min(values, key=lambda x: x[1])
        else:
            best_epoch, best_value = max(values, key=lambda x: x[1])
        
        return best_value, best_epoch


class TrainingLogger:
    """Comprehensive logger for training runs."""
    
    def __init__(self, output_dir: Path, experiment_name: str):
        """
        Initialize training logger.
        
        Args:
            output_dir: Output directory
            experiment_name: Name of experiment
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self.logger = setup_logger(
            experiment_name,
            log_file=self.output_dir / 'train.log'
        )
        
        self.metrics_logger = MetricsLogger(self.output_dir)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics."""
        self.metrics_logger.log(epoch, metrics)
        
        # Also log to console
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.info(f"Epoch {epoch} | {metrics_str}")
    
    def log_config(self, config: dict):
        """Log configuration."""
        config_file = self.output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.info(f"Configuration saved to {config_file}")
    
    def get_best_metric(self, metric: str, mode: str = 'min'):
        """Get best metric value."""
        return self.metrics_logger.get_best(metric, mode)


# Example usage
if __name__ == "__main__":
    # Test logging
    logger = setup_logger("test", use_colors=True)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test metrics logger
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_logger = MetricsLogger(Path(tmpdir))
        metrics_logger.log(1, {'loss': 0.5, 'accuracy': 0.8})
        metrics_logger.log(2, {'loss': 0.3, 'accuracy': 0.9})
        
        best_loss, epoch = metrics_logger.get_best('loss', 'min')
        print(f"Best loss: {best_loss} at epoch {epoch}")
