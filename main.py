from readline import parse_and_bind
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from loguru import logger
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from config import (
    API_PREFIX,
    DEBUG,
    ELK_ENABLED,
    APM_SERVER_TOKEN,
    APM_SERVER_URL,
    APM_SERVICE_NAME
)
from errors.http_error import http_error_handler
from errors.validation_error import (
    http422_error_handler,
)

from offensive import router
from elasticapm.contrib.starlette import make_apm_client, ElasticAPM

def get_application() -> FastAPI:
    application = FastAPI(title="Offensive Detector Service", debug=DEBUG)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if ELK_ENABLED == "true":
        apm_config = {
            'SERVICE_NAME': APM_SERVICE_NAME,
            'SERVER_URL': APM_SERVER_URL,
            'SECRET_TOKEN': APM_SERVER_TOKEN,
            'ENVIRONMENT': 'production',
            'CAPTURE_BODY':'all',
            'CAPTURE_HEADERS': True,
        }
        apm = make_apm_client(apm_config)
        application.add_middleware(ElasticAPM, client=apm)
    application.add_exception_handler(HTTPException, http_error_handler)
    application.add_exception_handler(RequestValidationError, http422_error_handler)


    application.include_router(router, prefix=API_PREFIX)

    return application


app = get_application()