# Import the necessary modules
import pickle
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

## Set up objects
# import model
with open(r"C:\Users\Spectra\Documents\GitHub\nd0821-c3-starter-code\starter\model\xgb_trained.pkl", "rb") as file_name:
    model = pickle.load(file_name)

# get encodings
with open(r"C:\Users\Spectra\Documents\GitHub\nd0821-c3-starter-code\starter\data\encodings.json") as json_file:
    encodings = json.load(json_file)

# Set up body
class Predictors(BaseModel):
    age : int
    fnlwgt : int
    education_num : int
    capital_gain : int
    capital_loss : int
    hours_per_week : int
    workclass : str
    education : str
    marital_status : str
    occupation : str
    relationship : str
    race : str
    sex : str
    native_country : str

# instantiate app
app = FastAPI()

# Create endpoints
@app.get("/")
def salutation():
    return 'Welcome to the Income Prediction Page'

@app.post("/predict-income")
def predict(data : Predictors):
    prediction_data = np.array([
        data.age,
        data.fnlwgt,
        data.education_num,
        data.capital_gain,
        data.capital_loss,
        data.hours_per_week,
        [int(key) for key, value in encodings['workclass'].items() if value == data.workclass][0],
        [int(key) for key, value in encodings['education'].items() if value == data.education][0],
        [int(key) for key, value in encodings['marital_status'].items() if value == data.marital_status][0],
        [int(key) for key, value in encodings['occupation'].items() if value == data.occupation][0],
        [int(key) for key, value in encodings['relationship'].items() if value == data.relationship][0],
        [int(key) for key, value in encodings['race'].items() if value == data.race][0],
        [int(key) for key, value in encodings['sex'].items() if value == data.sex][0],
        [int(key) for key, value in encodings['native_country'].items() if value == data.native_country][0]
    ]).reshape(1,-1)
    
    # Predict
    prediction = model.predict(prediction_data)[0]

    if prediction == 0:
        return {"Income": "Below or equal to 50K USD"}
    else:
        return {"Income": "Above 50K USD"}
