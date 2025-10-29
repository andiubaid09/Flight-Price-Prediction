import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path dataset berada di sini', path)
print('Isi dari path ini adalah', os.listdir(path))
df = pd.read_csv(path + '/airlines_flights_data.csv')
df
df.drop(columns='index', inplace=True)
df.head(5)
df.columns

Features = ['source_city','departure_time','stops','arrival_time',
            'destination_city','class','days_left']
X = df[Features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data training berjumlah :{len(X_train)} dan data uji berjumlah : {len(X_test)}")