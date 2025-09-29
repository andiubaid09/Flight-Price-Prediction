from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Memilah Fitur dan Target
Features = ['source_city','departure_time','stops', 'arrival_time',
            'destination_city','class','days_left']
fitur  = df[Features]
target = df['price']

# Membagi datasheet 
X_train, X_test, y_train, y_test = train_test_split(fitur,target,
                                                    test_size=0.3,
                                                    random_state=42)
print(f"Data latih:{len(X_train)} observasi dan Data test:{len(X_test)} observasi")

# Normalisasi/Standarisasi data kolom days_left
encoder = StandardScaler()
numeric_features = ['days_left']
numeric_transform = encoder

# Mengubah teks pada kolom class menjadi data numerik yang terurut 
ordinal_features = ['class']
class_categories = ['Economy','Business']
ordinal_transform = OrdinalEncoder(categories=[class_categories])

# Mengubah fitur-fitur yang tidak memiliki urutan dari teks menjadi representasi numerik binear (0 dan 1)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_categ = ['source_city','departure_time','stops','arrival_time','destination_city']
encoder_transform = encoder

# Gabungkan dalam ColumnTransfer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, numeric_features),
        ('ord', ordinal_transform, ordinal_features),
        ('cat', encoder_transform, encoder_categ)
    ],
    remainder='drop'
)

# Mengubah skala fitur harga atau fitur numerik lainnya menggunakan transformasi log
# Dilengkapi fungsi kebalikan untuk menginterpretasikan hasil prediksi
log_transform = FunctionTransformer(np.log1p, inverse_func=np.expm1)

# Membuat model RandomForest Regressor
rfr = RandomForestRegressor(random_state=42)

# Wrapper : Model yang membungkus rfr dan otomatis menangani Y
regressor_wrapper = TransformedTargetRegressor(
    regressor=rfr,
    func=log_transform.func,
    inverse_func=log_transform.inverse_func
)

# Pembangunan model pipeline Awal (Estimator untuk GridSearchCV)
rfr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor_wrapper)
])