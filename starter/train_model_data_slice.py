import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import inference, train_model, compute_model_metrics
import sys

df = pd.read_csv("data/clean_census.csv")

sys.stdout = open("slice_output.txt", "w")
# Train the model on slices of data based on race
for idx, val in enumerate(df.race.unique()):
    df_slice = df[df['race']==val]
    df_slice['race'] = df["race"].replace(val, idx)

    train, test = train_test_split(df_slice, test_size=0.20, random_state=0)

    #removed sex from cat_features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
    )

    model, acc = train_model(X_train, y_train)

    preds = inference(model=model, X=X_test)

    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
    print(
        f"Slice for: {val},\n"
        f"Accuracy: {acc},\n"
        f"Precision: {precision},\n"
        f"Recall: {recall},\n"
        f"Fbeta: {fbeta}")

sys.stdout.close()