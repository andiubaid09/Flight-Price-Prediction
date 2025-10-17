import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path Dataset berada di :', path)
print('Isi folder :', os.listdir(path))
df = pd.read_csv(path + '/airlines_flights_data.csv')
df
df.drop(columns='index', inplace=True)
df.head(5)
df.columns
Features = ['source_city', 'departure_time','stops','arrival_time','destination_city','class','days_left']
Fitur = df[Features]
Target = df['price']
X_train , X_test, y_train, y_test = train_test_split(
    Fitur, Target, test_size=0.2, random_state=42
)
print(f"Data Training berjumlah :{len(X_train)} dan Data Uji berjumlah : {len(X_test)}")
scaler = StandardScaler()
num_feat = ['days_left']
num_transform = scaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_feat = ['source_city','departure_time','stops','arrival_time','destination_city']
cat_transform = encoder
ordinal_features = ['class']
class_categories = ['Economy','Business']
ordinal_transform = OrdinalEncoder(categories=[class_categories])
log_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1)
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('log', log_transform),
        ('scaler', num_transform)]),num_feat),
    ('ord', ordinal_transform, ordinal_features),
    ('cat', cat_transform, cat_feat)
], remainder='drop')
model = Lasso()
pipeline_lasso = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', TransformedTargetRegressor(
        regressor= model,
        func= np.log1p,
        inverse_func=np.expm1
    ))
])
params_grid = {
    'regressor__regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}