import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline


path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path Dataset berada di :', path)
print('Isi folder :', os.listdir(path))
df.drop(columns='index', inplace=True)
df.head(5)
df.columns
Features = ['source_city','departure_time','stops','arrival_time','destination_city','class','days_left']
X = df[Features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)
print(f"Data training berjumlah {len(X_train)} dan Data Uji berjumlah {len(X_test)}")
scaler = StandardScaler()
num_features = ['days_left']
num_transform = scaler

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_features = ['source_city','departure_time','stops','arrival_time','destination_city']
cat_transform = encoder