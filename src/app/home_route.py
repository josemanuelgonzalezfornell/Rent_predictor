from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from sklearn.metrics import r2_score
import utils.functions as fn
import re


home_route = APIRouter()


templates = Jinja2Templates(directory="templates")



@home_route.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    model = joblib.load(open("model/model.pkl", "rb"))
    df = pd.read_csv("data/last_year_data.csv", sep=";", index_col=0)
    df.columns = [re.sub(r"_20\d{2}$", '', col) for col in df.columns]
    df_features = df.drop("Alquiler_mes_vu_m2", axis=1)
    r2score = round(r2_score(df["Alquiler_mes_vu_m2"], fn.get_prediction(model, df_features, "Comunidad_autonoma", ["Provincia", "Codigo_municipio","Codigo_provincia"])) * 100, 2)
    return templates.TemplateResponse("home.html", {"request": request, "r2score": r2score})