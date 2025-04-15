import logging
import sys
def setup_logging(level=logging.INFO):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(message)s',
        stream=sys.stdout
    )