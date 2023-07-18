from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from predict_indv_route import predict_indv_route
from home_route import home_route
from predict_csv_route import predict_csv_route
from retrain_route import retrain_route
from leyend_route import leyend_route

app = FastAPI()

app.mount("/images", StaticFiles(directory="img"), name="images")

app.include_router(home_route)

app.include_router(predict_indv_route)

app.include_router(predict_csv_route)

app.include_router(retrain_route)

app.include_router(leyend_route)
