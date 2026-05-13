# This file trains ML models and logs results to MLflow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import pickle
from dotenv import load_dotenv
import os

from preprocess import load_transformed_data, preprocess
from mlflow_tracker import log_run

load_dotenv()

def train():

    # load transformed data
    df = load_transformed_data()

    # preprocess data
    X, y = preprocess(df)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_acc = accuracy_score(
        y_test,
        rf_model.predict(X_test)
    )

    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train)

    xgb_acc = accuracy_score(
        y_test,
        xgb_model.predict(X_test)
    )

    print(f"XGBoost Accuracy: {xgb_acc:.4f}")

    # choose best model
    if xgb_acc >= rf_acc:

        best_model = xgb_model
        best_name = "XGBoost"

    else:

        best_model = rf_model
        best_name = "RandomForest"

    print(f"Best Model: {best_name}")

    # predictions
    y_pred = best_model.predict(X_test)

    y_prob = best_model.predict_proba(X_test)[:, 1]

    # metrics
    acc = accuracy_score(y_test, y_pred)

    pre = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ROC AUC
    if len(set(y_test)) > 1:

        auc = roc_auc_score(y_test, y_prob)

        print(f"ROC AUC Score: {auc:.4f}")

    else:

        auc = 0

        print("ROC AUC Score cannot be calculated (only one class present)")

    # log to mlflow
    log_run(
        best_model,
        best_name,
        acc,
        pre,
        rec,
        f1,
        auc
    )

    # save model
    os.makedirs('./ml', exist_ok=True)

    with open(os.getenv('MODEL_PATH'), 'wb') as f:

        pickle.dump(best_model, f)

    print(f"Model saved to {os.getenv('MODEL_PATH')}")

    return best_model, X_test, y_test, best_name


if __name__ == "__main__":

    train()