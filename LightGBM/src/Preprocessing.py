import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path berada di :', path)
print('Isi Path tersebut adalah', os.listdir(path))
df = pd.read_csv(path + '/airlines_flights_data.csv')
df
df.drop(columns='index',inplace=True)
df.head(5)
Features = ['source_city', 'departure_time','stops','arrival_time','destination_city','class','days_left']
X = df[Features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Jumlah data training :{len(X_train)} dan jumlah data uji :{len(X_test)}")
scaler = StandardScaler()
num_feat = ['days_left']
num_transform = scaler

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_feat = ['source_city','departure_time','stops','arrival_time','destination_city']
cat_transform = encoder

ordinal_feat = ['class']
class_categ = ['Economy','Business']
ordinalt_transform = OrdinalEncoder(categories=[class_categ])

log_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1)
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('log',log_transform),
        ('scaler', num_transform)
    ]), num_feat),
    ('ord', ordinalt_transform, ordinal_feat),
    ('cat', cat_transform, cat_feat)
], remainder='drop')
lbgm = LGBMRegressor(random_state=42)
model_lbgm = TransformedTargetRegressor(
    regressor = lbgm,
    func = np.log1p,
    inverse_func= np.expm1
)
lbgm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model_lbgm)
])