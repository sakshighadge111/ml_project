# This file extracts raw data from MySQL database

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def get_engine():

    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    return create_engine(url)

def extract_data():

    engine = get_engine()

    df = pd.read_sql("SELECT * FROM online_shoppers", engine)

    print(f"Extracted {len(df)} rows")

    return df

if __name__ == "__main__":

    df = extract_data()

    print(df.head())