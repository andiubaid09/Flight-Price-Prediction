import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print("Lokasi File Berada di Path berikut :", path)
print("Isi Path tersebut adalah", os.listdir(path))
df = pd.read_csv(path + '/airlines_flights_data.csv')
df
df.drop(columns = 'index', inplace=True)
df.head(5)
Features = ['source_city','departure_time','stops','arrival_time','destination_city','class','days_left']
X = df[Features]
y = df['price']