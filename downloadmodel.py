from loguru import logger
from config import MODEL_URL
import requests

async def download_and_save_offensive_model() -> bool:
    try:
        response = requests.get(MODEL_URL)
        open("model.h5", "wb").write(response.content)
        return True
    except Exception as e:
        logger.error(e.__str__)
