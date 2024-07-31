from fastapi import FastAPI
from pydantic import BaseModel
from calculator import calculate

app = FastAPI()


@app.get("/")
async def get_root_route():
    return {"message": "calc"}


class Formula(BaseModel):
    x: float
    y: float
    operator: str


@app.post("/calculator")
async def input_formula(input: Formula):
    return calculate(input.x, input.y, input.operator)
