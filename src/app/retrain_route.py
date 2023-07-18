from fastapi import APIRouter, Request, Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
from sklearn.metrics import r2_score
import utils.functions as fn
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import re

retrain_route = APIRouter()

templates = Jinja2Templates(directory="templates")

@retrain_route.get("/retrain", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse("retrain.html", {"request": request})

@retrain_route.post("/retrain", response_class=HTMLResponse)
async def retrain(request: Request, csvFile: UploadFile = Form(...)):
    content = csvFile.file.read()
    content_str = content_str = StringIO(content.decode('utf-8'))
    df = pd.read_csv(content_str, sep=";", index_col=0)

    df_deprecated =  pd.read_csv("data/current_year_data.csv", sep=";", index_col=0)
    df_deprecated.to_csv("data/last_year_data.csv", sep=";", index=True)

    df.to_csv("data/current_year_data.csv", sep=";", index=True)

    df.drop(["Provincia", "Codigo_municipio","Codigo_provincia"],axis=1, inplace=True)
    encoder = OrdinalEncoder()
    df[["Comunidad_autonoma"]] = encoder.fit_transform(df[["Comunidad_autonoma"]])
    df.columns = [re.sub(r"_20\d{2}$", '', col) for col in df.columns]

    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(["Alquiler_mes_vu_m2"], axis=1), df["Alquiler_mes_vu_m2"], test_size=0.2, random_state=42)
    random_forest = RandomForestRegressor(random_state=42)
    random_forest.fit(X_train, Y_train)
    df_features = pd.DataFrame({'features':X_train.columns, 'importances':random_forest.feature_importances_}).sort_values('importances', ascending=False).set_index("features")
    df_features.to_csv("data/df_features.csv", sep=";", index=True)

    X_train.drop(df_features[6:].index, axis=1, inplace=True)
    X_test.drop(df_features[6:].index, axis=1, inplace=True)
    df.drop(df_features[6:].index, axis=1, inplace=True)

    # model_train, model_train_score, test_metrics, final_model = fn.get_best_model(df, "Alquiler_mes_vu_m2",X_train, Y_train, X_test, Y_test, 5, "R2")

    model_train_score, test_metrics, final_model = fn.get_best_model(df, "Alquiler_mes_vu_m2",X_train, Y_train, X_test, Y_test, 5, "R2")
     
    joblib.dump(final_model, "model/model.pkl")

    return templates.TemplateResponse("retrain.html", {"request": request, "model_train_score": round(model_train_score, 2), "test_metrics": test_metrics})