import logging
import sys

def setup_logging(level=logging.INFO, error_log_path="error.log"):
    """
    Configure logging with console output and file logging for errors.

    Args:
        level (int): Logging level for console output (e.g., logging.INFO).
        error_log_path (str): Path to the file for error logs.
    """
    # Reset existing configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter

    # Console handler (e.g., INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler (ERROR and above)
    file_handler = logging.FileHandler(error_log_path)
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers to root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
