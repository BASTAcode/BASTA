import logging
import sys
from pathlib import Path


def setup_logger(logfile: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that writes both to a log file and the terminal.

    Parameters
    ----------
    logfile : str
        Path to the output log file. ".log" will be added if not provided.
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is INFO.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logfile = Path(logfile)
    if not logfile.suffix:
        logfile = logfile.with_suffix(".log")

    logger = logging.getLogger("basta_logger")
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging

    # Clear existing handlers (useful in interactive sessions)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class StreamToLogger:
    """
    Redirects print-style output to a logger.

    Example
    -------
    logger = setup_logger("output")
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    print("This goes to the logfile and terminal via the logger.")
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        pass
