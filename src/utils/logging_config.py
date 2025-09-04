"""
Centralized logging configuration to prevent duplicate handlers.
"""
import logging
import sys
from pathlib import Path

_logging_configured = False

def setup_logging(log_file: str = "helm_processor.log", logger_name: str = "HELM_Processor"):
    """
    Set up logging configuration once and return a logger.
    
    Args:
        log_file: Name of the log file
        logger_name: Name of the logger
    
    Returns:
        Logger instance
    """
    global _logging_configured
    
    # Only configure logging once
    if not _logging_configured:
        # Remove all existing handlers from the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure logging once
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        _logging_configured = True
    
    return logging.getLogger(logger_name)
