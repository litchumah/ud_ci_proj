# Put the code for your API here.
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
from starter.ml.data import process_data
from starter.ml.model import inference
from joblib import load
import pandas as pd
import os
# Start up dvc pull.
os.system("dvc config core.no_scm true")
os.system("dvc pull")
os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Person(BaseModel):
    age: int
    workclass: Literal['State-gov', 'Self-emp-not-inc', 'Private',
                       'Federal-gov','Local-gov', 'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                       'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
                       '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int = Field(alias="education-num")
    marital_status: Literal['Never-married', 'Married-civ-spouse', 'Divorced',
                            'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
                            'Widowed'] = Field(alias="marital-status")
    occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
                        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
                        'Craft-repair', 'Protective-serv', 'Armed-Forces',
                        'Priv-house-serv']
    relationship: Literal['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
                          'Other-relative']
    race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
                  'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
                            'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
                            'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
                            'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                            'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
                            'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
                            'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                            'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                            'Holand-Netherlands'] = Field(alias="native-country")

model=load('model/trained_model.joblib')
encoder=load('model/encoder.joblib')
lb=load('model/label_binarizer.joblib')

@app.get("/")
async def root():
    return {"message": "Welcome to the income classifier API!"}

@app.post("/inference")
async def predict_income(person: Person):
    data = pd.DataFrame.from_dict([person.dict(by_alias=True)])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, encoder=encoder, lb=lb, training=False
    )

    preds = inference(model, X)
    return {
        "prediction": float(preds[0])
    }
