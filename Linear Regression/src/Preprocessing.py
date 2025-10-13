import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path Dataset berada di :', path)
print('Isi folder :', os.listdir(path))
df = pd.read_csv(path + '/airlines_flights_data.csv')
df
Features = ['source_city','departure_time','stops','arrival_time','destination_city','class','days_left']
fitur = df[Features]
target = df['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    fitur, target, test_size = 0.2, random_state=42
)
print(f"Data latih berjumlah {len(X_train)} dan data uji observasi berjumlah {len(X_test)}")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer

scaler = StandardScaler()
numeric_features = ['days_left']
numeric_tranform = scaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_cat = ['source_city','departure_time','stops','arrival_time','destination_city']
encoder_transform = encoder
ordinal_features = ['class']
class_categories = ['Economy','Business']
ordinal_transform = OrdinalEncoder(categories=[class_categories])

import numpy as np

log_tranform = FunctionTransformer(np.log1p, inverse_func=np.expm1)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('log', log_tranform),
            ('scaler', scaler)]), numeric_features),
        ('ord', ordinal_transform, ordinal_features),
        ('cat', encoder_transform, encoder_cat)
    ], remainder = 'drop'
)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

from sklearn.compose import TransformedTargetRegressor

model_linear = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor = regressor,
        func = np.log1p,
        inverse_func = np.expm1
    ))
])