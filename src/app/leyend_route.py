from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import re

leyend_route = APIRouter()

templates = Jinja2Templates(directory="templates")

@leyend_route.get("/leyend", response_class=HTMLResponse)
async def get_leyend(request: Request):

    # read data
    df = pd.read_csv("data/current_year_data.csv", sep=";", index_col=0)
    patron = "^Alquiler_mes_vu_m2"

    # drop target column
    columnas_to_drop = [columna for columna in df.columns if re.match(patron, columna)]
    df.drop(columnas_to_drop, axis=1, inplace=True)

    # get Dataframe with columns
    columns = pd.DataFrame(df.columns, columns=[""])
    
    return templates.TemplateResponse("leyend.html", {"request": request, "columns": columns})