import logging
from logging.handlers import RotatingFileHandler
import coloredlogs

class CustomLogger:
    def __init__(self, name: str, log_gile: str):
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
        
            file_handler = RotatingFileHandler(log_gile, mode='a', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            file_formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | PID:%(process)d | TID:%(thread)d | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"
            )

            console_formatter = coloredlogs.ColoredFormatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # file_handler.setFormatter(FileColorFormatter())
            file_handler.setFormatter(file_formatter)
            # console_handler.setFormatter(ConsoleColorFormatter())
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return LoggerAdapter(self.logger)

class LoggerAdapter(logging.LoggerAdapter):

    def debug(self, msg: str, *args, **kwargs):
        super().debug(msg, *args, **{**kwargs, "stacklevel": 2})
    
    def info(self, msg: str, *args, **kwargs):
        super().info(msg, *args, **{**kwargs, "stacklevel": 2})
    
    def warning(self, msg: str, *args, **kwargs):
        super().warning(msg, *args, **{**kwargs, "stacklevel": 2})
    
    def error(self, msg: str, *args, **kwargs):
        super().error(msg, *args, **{**kwargs, "stacklevel": 2})
    
    def critical(self, msg: str, *args, **kwargs):
        super().critical(msg, *args, **{**kwargs, "stacklevel": 2})

class ConsoleColorFormatter(logging.Formatter):
    grey = "\x1b[90m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class FileColorFormatter(logging.Formatter):
    grey = "\x1b[90m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(name)s | %(levelname)s | PID:%(process)d | TID:%(thread)d | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)