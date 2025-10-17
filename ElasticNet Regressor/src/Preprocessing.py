import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path Dataset berada di :', path)
print('Isi Folder :', os.listdir(path))

df = pd.read_csv( path + '/airlines_flights_data.csv')
df
df.drop(columns='index', inplace=True)
df.head(5)
df.columns
Features = ['source_city','departure_time','stops','arrival_time','destination_city','class','days_left']
Fitur = df[Features]
Target = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    Fitur, Target, random_state=42, test_size=0.2
)
print(f'Data Training berjumlah : {len(X_train)} dan Data Uji berjumlah : {len(X_test)}')

scaler = StandardScaler()
num_features = ['days_left']
num_transform = scaler


