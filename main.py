import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = FastAPI()
model = joblib.load("model/model.pkl")


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    visit_date: str
    visit_time: str
    hit_date: str


class Prediction(BaseModel):
    Result: int


@app.get("/status")
def status():
    return "I'm OK"


@app.get("/version")
def version():
    return model["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(form: Form):

    predict_columns = [
        "visit_month",
        "visit_day_of_week",
        "hit_month",
        "hit_day_of_week",
        "utm_source_BHcvLfOaCWvWTykYqHVe",
        "utm_source_MvfHsxITijuriZxsqZqt",
        "utm_source_QxAxdyPLuQMEcrdZWdWb",
        "utm_source_RmEBuqrriAfAVsLQQmhk",
        "utm_source_ZpYIoDJMcFzVoPFsHGJL",
        "utm_source_aXQzDWsJuGXeBXexNHjc",
        "utm_source_bByPQxmDaMXgpHeypKSM",
        "utm_source_fDLlAcSmythWSCVMvqvL",
        "utm_source_jaSOmLICuBzCFqHfBdRg",
        "utm_source_kjsLglQLzykiRbcDiGcD",
        "utm_source_other",
        "utm_source_vFcAhRxLfOWKhvxjELkx",
        "device_category_desktop",
        "device_category_mobile",
        "device_category_tablet",
        "geo_country_Russia",
        "geo_country_other",
        "geo_city_Chelyabinsk",
        "geo_city_Kazan",
        "geo_city_Krasnodar",
        "geo_city_Krasnoyarsk",
        "geo_city_Moscow",
        "geo_city_Nizhny Novgorod",
        "geo_city_Novosibirsk",
        "geo_city_Saint Petersburg",
        "geo_city_Samara",
        "geo_city_Ufa",
        "geo_city_Yekaterinburg",
        "geo_city_other"   
        ]

    predict_df = pd.DataFrame(columns=[predict_columns])

    form_df = pd.DataFrame ({
        "visit_date": [form.dict()["visit_date"]],
        "visit_time": [form.dict()["visit_time"]],
        "hit_date": [form.dict()["hit_date"]]
        })

    form_df["visit_date"] = pd.to_datetime(form_df["visit_date"])
    form_df["hit_date"] = pd.to_datetime(form_df["hit_date"])

    predict_df["visit_month"] = form_df["visit_date"].apply(lambda x: x.month)
    predict_df["visit_day_of_week"] = form_df["visit_date"].dt.dayofweek
    predict_df["hit_month"] = form_df["hit_date"].apply(lambda x: x.month)
    predict_df["hit_day_of_week"] = form_df["hit_date"].dt.dayofweek

    utm_source = []
    device_category = []
    geo_country = []
    geo_city = []

    for elem in predict_columns:
        if elem.split("_")[0] == "utm":
            utm_source.append(elem)
        elif elem.split("_")[0] == "device":
            device_category.append(elem)
        elif elem.split("_")[1] == "country":
            geo_country.append(elem)
        elif elem.split("_")[1] == "city":
            geo_city.append(elem)

    columns_list = [utm_source, device_category, geo_country, geo_city]

    for column in columns_list:
        for elem in column:
            if elem.split("_")[2] == form.dict()[elem.split("_")[0]+"_"+elem.split("_")[1]]: 
                predict_df[elem] = 1
            if elem.split("_")[2] != form.dict()[elem.split("_")[0]+"_"+elem.split("_")[1]]: 
                predict_df[elem] = 0

    y = model["model"].predict(predict_df)

    return {
        "Result": y[0]
    }