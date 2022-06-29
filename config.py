import logging
import sys
import os
from loguru import logger
from starlette.config import Config
from logginghandler import InterceptHandler

config = Config(".env", os.environ)

DEBUG: bool = config("DEBUG", cast=bool, default=True)

API_PREFIX = "/api"
MODEL_URL: str = config("MODEL_URL", cast=str, default="")

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOGGERS = ("uvicorn.asgi", "uvicorn.access")

logging.getLogger().handlers = [InterceptHandler()]
for logger_name in LOGGERS:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler(level=LOGGING_LEVEL)]

logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])