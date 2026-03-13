# Final Project – ML Conversion Prediction API

## Install dependencies

pip install -r requirements.txt

## Train model

python src/train_model.py

## Run API

uvicorn app:app --reload

## Open API

http://127.0.0.1:8000/docs

## Project Description

This project builds a machine learning model that predicts the probability of a target event (user conversion) based on Google Analytics session data.

The project includes:

- Data preprocessing
- Feature engineering
- Model training using CatBoost
- Model evaluation
- REST API for predictions using FastAPI

The API allows sending session parameters and receiving the probability of conversion.


---

# Model Training

The model is trained using the CatBoost algorithm.

Features include:

- visit_number
- visit_hour
- visit_dayofweek
- visit_month
- is_returning_user
- utm_source
- utm_medium
- utm_campaign
- utm_adcontent
- utm_keyword
- device_category
- device_os
- device_brand
- device_browser
- geo_city
- device_screen_resolution

The target variable is: sub_submit_success


---

# Model Performance

Evaluation metric: ROC-AUC

Result: ROC-AUC = 0.733



