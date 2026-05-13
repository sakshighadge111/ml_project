# This file evaluates the saved model on test dataimport pickleimport numpy as npfrom sklearn.metrics import (    accuracy_score,    precision_score,    recall_score,    f1_score,    confusion_matrix,    roc_auc_score)from dotenv import load_dotenvimport osfrom preprocess import load_transformed_data, preprocessfrom sklearn.model_selection import train_test_splitload_dotenv()def evaluate():    # load saved model    with open(os.getenv('MODEL_PATH'), 'rb') as f:        model = pickle.load(f)    print("Model loaded successfully")    # load transformed data    df = load_transformed_data()    # preprocess data    X, y = preprocess(df)    # split data    _, X_test, _, y_test = train_test_split(        X,        y,        test_size=0.2,        random_state=42    )    # predictions    y_pred = model.predict(X_test)    # prediction probabilities    y_prob = model.predict_proba(X_test)[:, 1]    # metrics    acc = accuracy_score(y_test, y_pred)    pre = precision_score(        y_test,        y_pred,        zero_division=0    )    rec = recall_score(        y_test,        y_pred,        zero_division=0    )    f1 = f1_score(        y_test,        y_pred,        zero_division=0    )    # ROC AUC    if len(set(y_test)) > 1:        auc = roc_auc_score(y_test, y_prob)    else:        auc = 0        print("ROC AUC Score cannot be calculated (only one class present)")    # confusion matrix    cm = confusion_matrix(y_test, y_pred)    # print metrics    print(f"Accuracy  : {acc:.4f}")    print(f"Precision : {pre:.4f}")    print(f"Recall    : {rec:.4f}")    print(f"F1 Score  : {f1:.4f}")    print(f"ROC AUC   : {auc:.4f}")    print(f"Confusion Matrix:\n{cm}")    return acc, pre, rec, f1, aucif __name__ == "__main__":   
#  evaluate()import pickle
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from dotenv import load_dotenv
import os

from preprocess import load_transformed_data, preprocess

from sklearn.model_selection import train_test_split

load_dotenv()

def evaluate():

    # load saved model
    with open(os.getenv('MODEL_PATH'), 'rb') as f:

        model = pickle.load(f)

    print("Model loaded successfully")

    # load transformed data
    df = load_transformed_data()

    # preprocess data
    X, y = preprocess(df)

    # split data
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # predictions
    y_pred = model.predict(X_test)

    # prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

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

    else:

        auc = 0

        print("ROC AUC Score cannot be calculated (only one class present)")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # print metrics
    print(f"Accuracy  : {acc:.4f}")

    print(f"Precision : {pre:.4f}")

    print(f"Recall    : {rec:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC AUC   : {auc:.4f}")

    print(f"Confusion Matrix:\n{cm}")

    return acc, pre, rec, f1, auc


if __name__ == "__main__":

    evaluate()