from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float

    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int

    InternetService_Fiber_optic: int
    InternetService_No: int

    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int

    Contract_One_year: int
    Contract_Two_year: int
    PaperlessBilling_Yes: int

    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
def predict(data: CustomerData):
    raw = data.model_dump()

    input_renamed = {
        'SeniorCitizen': raw['SeniorCitizen'],
        'tenure': raw['tenure'],
        'MonthlyCharges': raw['MonthlyCharges'],
        'TotalCharges': raw['TotalCharges'],
        'gender_Male': raw['gender_Male'],
        'Partner_Yes': raw['Partner_Yes'],
        'Dependents_Yes': raw['Dependents_Yes'],
        'PhoneService_Yes': raw['PhoneService_Yes'],
        'MultipleLines_No phone service': raw['MultipleLines_No_phone_service'],
        'MultipleLines_Yes': raw['MultipleLines_Yes'],
        'InternetService_Fiber optic': raw['InternetService_Fiber_optic'],
        'InternetService_No': raw['InternetService_No'],
        'OnlineSecurity_No internet service': raw['OnlineSecurity_No_internet_service'],
        'OnlineSecurity_Yes': raw['OnlineSecurity_Yes'],
        'OnlineBackup_No internet service': raw['OnlineBackup_No_internet_service'],
        'OnlineBackup_Yes': raw['OnlineBackup_Yes'],
        'DeviceProtection_No internet service': raw['DeviceProtection_No_internet_service'],
        'DeviceProtection_Yes': raw['DeviceProtection_Yes'],
        'TechSupport_No internet service': raw['TechSupport_No_internet_service'],
        'TechSupport_Yes': raw['TechSupport_Yes'],
        'StreamingTV_No internet service': raw['StreamingTV_No_internet_service'],
        'StreamingTV_Yes': raw['StreamingTV_Yes'],
        'StreamingMovies_No internet service': raw['StreamingMovies_No_internet_service'],
        'StreamingMovies_Yes': raw['StreamingMovies_Yes'],
        'Contract_One year': raw['Contract_One_year'],
        'Contract_Two year': raw['Contract_Two_year'],
        'PaperlessBilling_Yes': raw['PaperlessBilling_Yes'],
        'PaymentMethod_Credit card (automatic)': raw['PaymentMethod_Credit_card_automatic'],
        'PaymentMethod_Electronic check': raw['PaymentMethod_Electronic_check'],
        'PaymentMethod_Mailed check': raw['PaymentMethod_Mailed_check']
    }

    df = pd.DataFrame([input_renamed])

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    result = "Churn" if prediction == 1 else "No Churn"

    return {"prediction":result}
