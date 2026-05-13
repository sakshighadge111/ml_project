from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
import os

sys.path.append(os.path.abspath("."))

from ml.predict import predict

app = FastAPI()

# static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# templates
templates = Jinja2Templates(directory="api/templates")


# request schema
class ShopperData(BaseModel):

    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: int
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: int
    Weekend: int


# home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# prediction API
@app.post("/predict")
async def make_prediction(data: ShopperData):

    input_data = data.dict()

    result = predict(input_data)

    return result