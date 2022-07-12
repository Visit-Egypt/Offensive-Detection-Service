import logging
import sys
import os
from loguru import logger
from starlette.config import Config
from logginghandler import InterceptHandler

config = Config(".env", os.environ)

DEBUG: bool = config("DEBUG", cast=bool, default=True)

API_PREFIX = "/api"
MODEL1_URL: str = config("MODEL1_URL", cast=str, default="")
MODEL2_URL: str = config("MODEL2_URL", cast=str, default="")
# ELK
APM_SERVER_URL: str = config("APM_SERVER_URL", cast = str, default="")
APM_SERVER_TOKEN: str = config("APM_SERVER_TOKEN", cast = str, default="")
APM_SERVICE_NAME: str = config("APM_SERVICE_NAME", cast = str, default="")
ELK_ENABLED: str = config("ELK_ENABLED", cast = str, default="false")

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOGGERS = ("uvicorn.asgi", "uvicorn.access")

logging.getLogger().handlers = [InterceptHandler()]
for logger_name in LOGGERS:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler(level=LOGGING_LEVEL)]

logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])