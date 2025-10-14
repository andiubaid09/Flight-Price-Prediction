from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as plt
import seaborn as sns

print("\n Best alpha:", grid_ridge.best_params_)
print("Best R2 Score (CV) :", grid_ridge.best_score_)
print(best_model)
y_pred = best_model.predict(X_test)
print('\n -- Hasil Evaluasi pada Data Uji --')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE (Root Mean Squared Error) : {rmse:.4f}')
print(f'MAE (Mean Absolute Error) : {mae:.4f}')
print(f'R-Squared (R2 Score) : {r2:.4f}')
print(f'Harga Aktual pada Data Uji : \n {y_test.values}')
print(f'Harga Prediksi dalam Skala Asli : \n{np.round(y_pred,2)}')

# Visualisasi Harga Aktual vs Harga Prediksi
seaborn_version = 'seaborn-v0_8-whitegrid'

plt.figure(figsize=(8,8))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', alpha=0.7)
max_val = max(y_test.max(), y_pred.max())
y_pred.max()

plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Garis prediksi sempurna (y=x)')
plt.title('Prediksi vs Nilai Aktual', fontsize=16, fontweight='bold')
plt.xlabel('Harga aktual', fontsize=12)
plt.ylabel('Harga Prediksi', fontsize=12)
plt.legend()
plt.text(0.05, 0.9, f'R2:{r2:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top')
plt.show()

# Visualisasi Distribusi Residual Error
residual = y_test - y_pred

plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.histplot(residual, kde=True, color='coral', bins=30)
plt.title('Distribusi Redisual Error', fontsize=16, fontweight='bold')
plt.xlabel('Residual (Harga Aktual - Harga Prediksi)', fontsize=12)
plt.axvline(0, color='r', linestyle='--', label='Pusat di Nole')
plt.legend()
plt.show()

# Visualisasi Residual vs Nilai Prediksi
plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_pred, y=residual, color='forestgreen', alpha=0.6)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.title('Residual vs Nilai Prediksi', fontsize=16, fontweight='bold')
plt.xlabel('Harga Prediksi', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.show()

# --- Fungsi untuk ambil nama fitur dari preprocessor ---
def get_feature_names_from_column_transformer(ct):
    feature_names = []

    for name, trans, cols in ct.transformers_:
        if name == 'remainder':
            continue

        if hasattr(trans, 'steps'):
            last_step = trans.steps[-1][1]
            if hasattr(last_step, 'get_feature_names_out'):
                names = last_step.get_feature_names_out(cols)
            else:
                names = cols
        elif hasattr(trans, 'get_feature_names_out'):
            names = trans.get_feature_names_out(cols)
        else:
            names = cols

        feature_names.extend(names)

    return feature_names

# --- Ambil nama fitur dan koefisien dari pipeline ---
preprocessor = best_model.named_steps['preprocessor']
nama_fitur = get_feature_names_from_column_transformer(preprocessor)

# Ambil koefisien dari model Linear Regression di dalam TransformedTargetRegressor
koef = best_model.named_steps['regressor'].regressor_.coef_

# --- Buat DataFrame Feature Importance ---
df_importance = pd.DataFrame({
    'Feature': nama_fitur,
    'Coefficient': koef
}).sort_values(by='Coefficient', ascending=False)

print(df_importance.head(20))

# Visualisasi Feature Importances
plt.figure(figsize=(10,6))
plt.barh(df_importance['Feature'], df_importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.gca().invert_yaxis()
plt.show()
