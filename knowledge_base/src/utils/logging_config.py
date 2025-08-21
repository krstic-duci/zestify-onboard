import logging


def setup_logging(level=logging.INFO):
    """
    Set up root logger to print filename and line number for all logs.
    Safe to call multiple times; avoids duplicate handlers.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    root_logger.setLevel(level)
