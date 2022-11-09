import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.ml.model import train_model, inference
from sklearn.model_selection import train_test_split

@pytest.fixture()
def data():
    """
    Load the dataset used for the analysis
    Returns
    -------
    data: pd.DataFrame
    """
    # Add code to load in the data.
    data = pd.read_csv("data/clean_census.csv")

    return data

def test_process_data(data):

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

    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape[1] == 104, f"X_train is expected to have 104 features."
    #this is due to not splitting the data
    assert y_train.shape[0] == 30162, f"y_train is expected to have 24129 rows."
    

def test_train_model(data):
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    results = model.predict(X_train)
    assert results.shape[0] == y_train.shape[0], "The number of labels and results are different!"

def test_scores(data):
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
    )

    model = train_model(X_train, y_train)
    scores = inference(model=model, X=X_test)
    assert scores.shape[0] == y_test.shape[0], "The number of labels and scores are different!"
