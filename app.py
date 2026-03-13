from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Путь до сохраненной обученной модели
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Путь до файла со списком признаков
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# загрузка модели и списка признаков
model = joblib.load(MODEL_PATH)

# загружаем список признаков, который был сохранён во время обучения. нужен для гарантии корректного порядка колонок
# и не забыть ни один признак
feature_cols = joblib.load(FEATURES_PATH)

# создание FASTAPI-приложения. Это основной объект API-сервиса. Именно его запускает uvicorn командой вида:
# uvicorn app:app --reload
app = FastAPI(
    title="Final Project API",
    version="1.0"
)

# описание формата входного запроса
# испольуя Pydantic-модель описываем структуру JSON, который клиент должен отправить в /predict
class VisitFeatures(BaseModel):
    # Числовые признаки визита
    visit_number: int
    visit_hour: int
    visit_dayofweek: int
    visit_month: int
    is_returning_user: int

    # Маркетинговые признаки
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str

    # Признаки устройства и географии
    device_category: str
    device_os: str
    device_brand: str
    device_browser: str
    geo_city: str
    device_screen_resolution: str

# служебный endpoint. Простой тестовый маршрут.
@app.get("/")
def root():
    return {"message": "API is running"}


# основной ENDPOINT для предсказания. Этот маршрут принимает POST-запрос с JSON, преобразует его в DataFrame, подаёт в модель
# и возвращает: prediction: 0 или 1; probability: вероятность целевого события
@app.post("/predict")
def predict(features: VisitFeatures):
    # Превращаем входной JSON в словарь
    # features приходит как объект Pydantic. model_dump() превращает его в обычный dict.
    payload = features.model_dump()

    # Превращаем словарь в pandas DataFrame. Модель CatBoost ожидает таблицу признаков,
    # поэтому даже один запрос оборачиваем в DataFrame из одной строки.
    data = pd.DataFrame([payload])

    # Выставляем правильный порядок колонок. Даже если признаки пришли в другом порядке,модель должна увидеть их ровно так,
    #  как во время обучения
    data = data[feature_cols]

    # Считаем вероятность класса 1. predict_proba возвращает вероятности двух классов:
    # [:, 0] -> вероятность класса 0
    # [:, 1] -> вероятность класса 1
    # Нам нужна вероятность совершения целевого действия,то есть вероятность класса 1
    proba = float(model.predict_proba(data)[0, 1])

    # Переводим вероятность в бинарный прогноз. Используем простой порог 0.5:
    # если вероятность >= 0.5 -> prediction = 1. иначе prediction = 0
    pred = int(proba >= 0.5)

    # Возвращаем результат в JSON. probability округляем до 6 знаков для аккуратного вывода
    return {
        "prediction": pred,
        "probability": round(proba, 6)
    }