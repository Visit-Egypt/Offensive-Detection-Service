from loguru import logger
from config import MODEL1_URL
from config import MODEL2_URL
import requests

async def download_and_save_offensive_model() -> bool:
    try:
        response1 = requests.get(MODEL1_URL)
        open("classifier2_jlib", "wb").write(response1.content)
        response2 = requests.get(MODEL2_URL)
        open("vectroize2_jlib", "wb").write(response2.content)
        return True
    except Exception as e:
        logger.error(e.__str__)
