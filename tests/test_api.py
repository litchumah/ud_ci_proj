from fastapi.testclient import TestClient
import os
from main import app

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the income classifier API!"}


def test_inference_less_than_50k():
    response = client.post("/inference", json={
                            "age": 39,
                            "workclass": "State-gov",
                            "fnlgt": 77516,
                            "education": "Bachelors",
                            "education-num": 13,
                            "marital-status": "Never-married",
                            "occupation": "Adm-clerical",
                            "relationship": "Not-in-family",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 2174,
                            "capital-loss": 0,
                            "hours-per-week": 40,
                            "native-country": "United-States"
                            })

    assert response.status_code == 200
    assert response.json() == {
            "prediction": 0
            }


def test_inference_more_than_50k():
    response = client.post("/inference", json={
                            "age": 31,
                            "workclass": "Private",
                            "fnlgt": 45781,
                            "education": "Masters",
                            "education-num": 14,
                            "marital-status": "Never-married",
                            "occupation": "Prof-specialty",
                            "relationship": "Not-in-family",
                            "race": "White",
                            "sex": "Female",
                            "capital-gain": 14084,
                            "capital-loss": 0,
                            "hours-per-week": 50,
                            "native-country": "United-States"
                            })
    assert response.status_code == 200
    assert response.json() == {
            "prediction": 1
            }
