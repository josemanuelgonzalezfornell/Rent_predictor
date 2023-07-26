from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import utils.functions as fn
import re


predict_indv_route = APIRouter()
templates = Jinja2Templates(directory="templates")


@predict_indv_route.get("/indv-predict", response_class=HTMLResponse)
async def welcome(request: Request):
    # read data current year
    df = pd.read_csv("data/current_year_data.csv", sep=";", index_col=0)

    # Clean dataframe current year
    df = fn.clean_df(df, "Comunidad_autonoma", ["Provincia", "Codigo_municipio","Codigo_provincia", "Alquiler_mes_vu_m2"])

    # Dicctionary to show the names of the columns more clearly
    traduccion = {"Total_vc": "Viviendas colectivas totales","Total_vu": "Viviendas unifamiliares totales","Alquiler_mes_vc_m2": "Precio de viviendas colectivas mes (m2)",
                  "Alquiler_mes_vu_m2": "Precio de viviendas unifamiliares mes (m2)", "Comunidad_autonoma": "Comunidad autonoma", "Poblacion": "Poblacion",
                  "Inmuebles_totales": "Inmuebles totales", "Viviendas_turisticas": "Viviendas turísticas totales", "Turistas": "Turistas anuales",
                  "Porcentaje_viviendas_turisticas": "Porcentaje de viviendas turísticas sobre el total de inmuebles en alquiler", "Total_casas_alquiler": "inmuebles en alquiler totales",
                  "Porcentaje_viviendas_alquiler": "Porcentaje de viviendas en alquiler sobre el total de inmuebles en alquiler",
                   "Porcentaje_vc_alquiler": "Porcentaje de viviendas colectivas en alquiler sobre el total de inmuebles en alquiler", "Porcentaje_vu_alquiler": "Porcentaje de viviendas unifamiliares en alquiler sobre el total de inmuebles en alquiler"}
    return templates.TemplateResponse("predict_indv.html", {"request": request, "df": df, "traduccion": traduccion})


@predict_indv_route.post("/indv-predict", response_class=HTMLResponse)
async def obtain_data(request: Request, Col1: str = Form(...), Col2: str = Form(...), Col3: str = Form(...), Col4: str = Form(...), Col5: str = Form(...), Col6: str = Form(...)):
    # read data current year
    df = pd.read_csv("data/last_year_data.csv", sep=";", index_col=0)
    
    # Clean dataframe current year
    df = fn.clean_df(df,"Comunidad_autonoma", ["Provincia", "Codigo_municipio","Codigo_provincia", "Alquiler_mes_vu_m2"], not_encoder=True)
    
    # Dicctionary to show the names of the columns more clearly
    traduccion = {"Total_vc": "Viviendas colectivas totales","Total_vu": "Viviendas unifamiliares totales","Alquiler_mes_vc_m2": "Precio de viviendas colectivas mes (m2)",
                  "Alquiler_mes_vu_m2": "Precio de viviendas unifamiliares mes (m2)", "Comunidad_autonoma": "Comunidad autonoma", "Poblacion": "Poblacion",
                  "Inmuebles_totales": "Inmuebles totales", "Viviendas_turisticas": "Viviendas turísticas totales", "Turistas": "Turistas anuales",
                  "Porcentaje_viviendas_turisticas": "Porcentaje de viviendas turísticas sobre el total de inmuebles en alquiler", "Total_casas_alquiler": "inmuebles en alquiler totales",
                  "Porcentaje_viviendas_alquiler": "Porcentaje de viviendas en alquiler sobre el total de inmuebles en alquiler",
                   "Porcentaje_vc_alquiler": "Porcentaje de viviendas colectivas en alquiler sobre el total de inmuebles en alquiler", "Porcentaje_vu_alquiler": "Porcentaje de viviendas unifamiliares en alquiler sobre el total de inmuebles en alquiler"}
    
    # Get de dtypes of the columns
    df_types = df.dtypes

    # Format the columns data
    data = pd.DataFrame([[df_types[0].type(Col1), df_types[1].type(Col2), df_types[2].type(Col3),
             df_types[3].type(Col4), df_types[4].type(Col5), df_types[5].type(Col6)]], columns=df.columns)
    
    # Load the model
    model = joblib.load("model/model.pkl")

    # Predict and print the prediction
    prediction = fn.get_prediction(model, data, "Comunidad_autonoma", cleaning=False)
    return templates.TemplateResponse("predict_indv.html", {"request": request, "prediction": round(prediction[0], 2), "df": df, "traduccion": traduccion})
