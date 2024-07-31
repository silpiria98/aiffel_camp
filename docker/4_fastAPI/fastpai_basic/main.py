from enum import Enum
from fastapi import FastAPI
from typing import List, Optional, Union
from datetime import datetime

# 데이터 타입 검증
from pydantic import BaseModel, Field

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/", include_in_schema=False)  # docs에 안나옴
def test():
    return {"asd": "test"}


@app.get("/home/{name}")
def health_check_handler(name: str):
    return {"name": name}


items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Buz"}]


@app.get("/items/{item_db}")
def read_item(item_db: str, skip: int = 0, limit: int = 10):
    return item_db[skip : skip + limit]


@app.post("/")
def home_post(msg: str):
    return {"Hello": "Post", "message": msg}


from fastapi import APIRouter

router = APIRouter()


@router.get("/hello")
async def say_hello() -> dict:
    return {"message": "Hello"}


# 분기점
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "alex"}

    if model_name is ModelName.resnet:
        return {"model_name": model_name, "message": "res"}


class Movie(BaseModel):
    mid: int
    genre: str
    rate: Union[int, float]
    tag: Optional[str] = None
    date: Optional[datetime] = None


class User(BaseModel):
    uid: int
    name: str = Field(min_length=2, max_length=7)
    age: int = Field(gt=1, le=130)


tmp_data = {"mid": 1, "genre": "action", "rate": 1.5}
tmp_user = {"uid": 100, "name": "asd", "age": 12}

tmp_movie = Movie(**tmp_data)
tmp_user_data = User(**tmp_user)
