# This file handles feature scaling, feature selection
# and class imbalance before training

import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import pickle
from dotenv import load_dotenv
import os

load_dotenv()


def get_engine():

    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    return create_engine(url)


def load_transformed_data():

    engine = get_engine()

    # LOAD CORRECT TABLE
    df = pd.read_sql(
        "SELECT * FROM online_shoppers",
        engine
    )

    print(f"Loaded {len(df)} rows")

    return df


def preprocess(df):

    # Convert Revenue TRUE/FALSE into 1/0
    df['Revenue'] = df['Revenue'].map({
        'TRUE': 1,
        'FALSE': 0,
        True: 1,
        False: 0
    })

    # Convert Weekend TRUE/FALSE into 1/0
    df['Weekend'] = df['Weekend'].map({
        'TRUE': 1,
        'FALSE': 0,
        True: 1,
        False: 0
    })

    # Convert Month text into numbers
    month_map = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'June': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

    df['Month'] = df['Month'].map(month_map)

    # Convert VisitorType text into numbers
    visitor_map = {
        'Returning_Visitor': 2,
        'New_Visitor': 0,
        'Other': 1
    }

    df['VisitorType'] = df['VisitorType'].map(visitor_map)

    # Fill missing values
    df = df.fillna(0)

    # Features and target
    X = df.drop('Revenue', axis=1)

    y = df['Revenue']

    # Revenue counts
    print(df['Revenue'].value_counts())

    print(
        f"Before SMOTE — Revenue 0: {(y==0).sum()}, Revenue 1: {(y==1).sum()}"
    )

    # Scale features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Save scaler
    with open('./ml/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Feature selection
    selector = SelectKBest(
        score_func=f_classif,
        k=10
    )

    X_selected = selector.fit_transform(X_scaled, y)

    # Selected feature names
    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    print(f"Selected Features: {selected_features}")

    # Apply SMOTE
    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_selected,
        y
    )

    print(
        f"After SMOTE — Revenue 0: {(y_resampled==0).sum()}, Revenue 1: {(y_resampled==1).sum()}"
    )

    # Save selector
    with open('./ml/selector.pkl', 'wb') as f:
        pickle.dump(selector, f)

    # Save selected features
    with open('./ml/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)

    print("Scaler, Selector saved successfully")

    return X_resampled, y_resampled


if __name__ == "__main__":

    df = load_transformed_data()

    X, y = preprocess(df)

    print(f"Final shape after preprocessing: {X.shape}")