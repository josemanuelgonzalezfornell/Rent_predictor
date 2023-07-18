from fastapi import APIRouter, Request, Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from io import StringIO
import utils.functions as fn

predict_csv_route = APIRouter()

templates = Jinja2Templates(directory="templates")

predict_csv_route.mount("/static", StaticFiles(directory="data"), name="static")

@predict_csv_route.get("/csv-predict", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse("predict_csv.html", {"request": request})

@predict_csv_route.post("/csv-predict", response_class=HTMLResponse)
async def predict_csv(request: Request, csvFile: UploadFile = Form(...)):
    model = joblib.load("model/model.pkl")
    content = csvFile.file.read()
    content_str= content_str = StringIO(content.decode('utf-8'))
    df = pd.read_csv(content_str, sep=";", index_col=0)  # Cargar el contenido del archivo en un DataFrame de pandas
    prediction = pd.DataFrame(fn.get_prediction(model, df, "Comunidad_autonoma", ["Provincia", "Codigo_municipio", "Codigo_provincia"]),index=df.index, columns=["Alquiler_mes_vu_m2"])
    prediction_str = prediction.to_string(index=True)
    return templates.TemplateResponse("predict_csv.html", {"request": request, "prediction": prediction_str})

