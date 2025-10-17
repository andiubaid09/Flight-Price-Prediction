from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib as plt
import seaborn as sns

print('Best Params yang ditemukan adalah :', grid_elastic.best_params_)
print('Best CV Scoring yang ditemukan adalah :', grid_elastic.best_score_)
print(best_elastic)
y_pred = best_elastic.predict(X_test)

print('=== HASIL EVALUASI MODEL ELASTIC ===')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Hasil rmse model Elastic adalah : {rmse:.4f}')
print(f'Hasil MAE model Elastic adalah : {mae:.4f}')
print(f'Hasil r2 model Elastic adalah : {r2:.4f}')
print(f'Harga Aktual Pada Data Uji : \n {y_test.values}')
print(f'Harga Prediksi Model : \n {np.round(y_pred)}')

# Visualisasi Harga Prediksi vs Harga Aktual
seaborn_version = 'seaborn-v0_8-whitegrid'

plt.figure(figsize=(8,8))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', alpha=0.7)
max_val = max(y_test.max(), y_pred.max())
y_pred.max()

plt.plot([0, max_val],[0, max_val], 'r--', lw=2, label='Garis prediksi sempurna (y=x)')
plt.title('Prediksi vs Nilai Aktual', fontsize=16, fontweight='bold')
plt.xlabel('Harga Aktual',fontsize=12)
plt.ylabel('Harga Prediksi', fontsize=12)
plt.legend()
plt.text(0.05, 0.9, f'R2{r2:.4f}', transform=plt.gca(). transAxes, fontsize=12, verticalalignment='top')
plt.show()

# Visualisasi Distribusi Residual Error
residual = y_test - y_pred

plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.histplot(residual, kde=True, color='coral', bins=30)
plt.title('Distribusi Residual Error', fontsize=16, fontweight='bold')
plt.xlabel('Residual (Harga Aktual - Harga Prediksi)', fontsize=12)
plt.axvline(0, color='r', linestyle='--', label='Pusat di Nole')
plt.legend()
plt.show()

# Visualisasi Residual Error vs Harga Prediksi
plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_pred, y=residual, color='forestgreen', alpha=0.6)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.title('Residual vs Nilai Prediksi', fontsize=16, fontweight='bold')
plt.xlabel('Harga Prediksi', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.show()

# Mengambil langkah preprocessornya dan regressornya

preprocessor = best_elastic.named_steps['preprocessor']
model = best_elastic.named_steps['regressor'].regressor_

feature_name = []

# Numeric StandardScaler preprocessor
num_fitur = preprocessor.transformers_[0][2]
feature_name.extend(num_fitur)

# Ordinal preprocessor
ordinal_fitur = preprocessor.transformers_[1][2]
feature_name.extend(ordinal_fitur)

# OneHotEncoder preprocessor
cat_fitur = preprocessor.transformers_[2][1]
cat_feat = cat_fitur.get_feature_names_out(preprocessor.transformers_[2][2])
feature_name.extend(cat_feat)

# Buat DataFrame

koefisien = model.coef_
feature_importances = pd.DataFrame({
    'Feature': feature_name,
    'Coeficient': koefisien
})

feature_importances['Abs_Coefficient'] = feature_importances['Coeficient'].abs()
feature_importances = feature_importances.sort_values('Abs_Coefficient', ascending=False)
feature_importances.head(10)

# Visualisasi Feature Importances
plt.figure(figsize=(10,6))
plt.barh(feature_importances['Feature'], feature_importances['Coeficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (ElasticNet Regression Coefficient)')
plt.gca().invert_yaxis()
plt.show()