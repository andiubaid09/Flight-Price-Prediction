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

scaler = StandardScaler()
num_feat = ['days_left']
num_transform = scaler

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_feat = ['source_city','departure_time','stops','arrival_time','destination_city']
cat_transform = encoder

df['class'].unique()
ordinal_features = ['class']
class_categories = ['Economy','Business']
ordinal_transform = OrdinalEncoder(categories=[class_categories])

log_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1)
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('log', log_transform),
        ('scaler', num_transform)
    ]), num_feat),
    ('ord', ordinal_transform, ordinal_features),
    ('cat', cat_transform, cat_feat)
], remainder='drop')

XGB = XGBRegressor(objective='reg:squarederror', random_state=42)
model_XGB = TransformedTargetRegressor(
    regressor = XGB,
    func = np.log1p,
    inverse_func= np.expm1
)
XGB_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model_XGB)
])