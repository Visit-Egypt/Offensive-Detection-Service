from enum import Enum

from fastapi import status
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
import pandas as pd 
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import  joblib
from loguru import logger
from downloadmodel import download_and_save_offensive_model


router = APIRouter()

class StatusEnum(str, Enum):
    OK = "OK"
    FAILURE = "FAILURE"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class HealthCheck(BaseModel):
    title: str = Field(..., description="API title")
    description: str = Field(..., description="Brief description of the API")
    version: str = Field(..., description="API server version number")
    status: StatusEnum = Field(..., description="API current status")


@router.get(
    "/status",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    tags=["Health Check"],
    summary="Performs health check",
    description="Performs health check and returns information about running service.",
)
def health_check():
    return {
        "title": "Offensive Detector Service",
        "description": "This is a test desc",
        "version": "0.0.0",
        "status": StatusEnum.OK,
    }

class Data(BaseModel):
    id: int
    comment_txt: str

class isOffensive(BaseModel):
    is_offensive: bool
    item_id: int

nltk.download('stopwords')
sn = SnowballStemmer(language='english')

classifier, word_vectorizer = None, None

@router.on_event("startup")
async def downloading_offensive_model() -> bool:
    logger.info("Downloading Offensive Detector Model ........")
    try:

        result = await download_and_save_offensive_model()
        if result:
            logger.info("Model Has Been Downloaded and Saved ...")
            return True
    except Exception as e:
        logger.exception(e.__str__)


@router.on_event("startup")
def load_model():
    classifier = joblib.load('classifier2_jlib')
    word_vectorizer = joblib.load('vectroize2_jlib')



def  clean_text(text):
    text =  text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    
    return text


def stemmer(text):
    words =  text.split()
    train = [sn.stem(word) for word in words if not word in set(stopwords.words('english'))]
    return ' '.join(train)



@router.post('/predict', response_model = isOffensive )
def make_test_predictions(comment: Data):
    df = {'id':comment.id,'comment_text':comment.comment_txt}
    df =  pd.DataFrame(df,index =[0])
    df.comment_text = df.comment_text.apply(clean_text)
    df.comment_text = df.comment_text.apply(stemmer)
    X_test = df.comment_text
    X_test_transformed = word_vectorizer.transform(X_test)
    y_test_pred = classifier.predict_proba(X_test_transformed)
    result =  sum(y_test_pred[0])
    if result >=1 :
        return isOffensive(is_offensive = True, item_id = comment.id) # {"Offensive Comment":"Delete IT"}

    else :
        return isOffensive(is_offensive = False, item_id = comment.id) #{"Normal  Comment": "No Action"}