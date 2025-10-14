import kaggelhub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

path = kagglehub.dataset_download('rohitgrewal/airlines-flights-data')
print('Path Dataset berada di :', path)
print('Isi Folder :', os.listdir(path))
df = pd.read_csv(path + '/airlines_flights_data.csv')
df
df.drop(columns='index', inplace=True)
df.head(5)
df.columns
Features = ['source_city','departure_time','stops','arrival_time','destination_city','class','days_left']
fitur = df[Features]
target = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    fitur, target, test_size=0.2, random_state=42
)
print(f"Data training berjumlah {len(X_train)} dan data uji berjumlah {len(X_test)}")

scaler = StandardScaler()
num_feat = ['days_left']
num_transform = scaler
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_feat = ['source_city','departure_time','stops','arrival_time','destination_city']
cat_transform = encoder
ordinal_features = ['class']
class_categories = ['Economy','Business']
ordinal_transofrm = OrdinalEncoder(categories=[class_categories])
log_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('log', log_transform),
            ('scaler', scaler)]), num_feat),
        ('ord', ordinal_transofrm, ordinal_features),
        ('cat', cat_transform, cat_feat)
    ], remainder='drop'
)
