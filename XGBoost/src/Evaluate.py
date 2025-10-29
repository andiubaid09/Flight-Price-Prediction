from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as plt
import seaborn as sns

print("Best Params yang ditemukan adalah : ", gridsearch_XGB.best_params_)
print("Best CV Score yang ditemukan adalah :", gridsearch_XGB.best_score_)
print(best_XGB)
y_pred = best_XGB.predict(X_test)

print('=== Evaluasi Model ===')
r2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

print(f'Best r2 socre yang dihasilkan adalah : {r2:.4f}')
print(f'Best MAE score yang dihasilkan adalah : {MAE:.4f}')
print(f'Best RMSE score yang dihasilkan adalah : {RMSE:.4f}')

# Visualisasi Harga Prediksi vs Harga Aktual
seaborn_version = 'seaborn-v0_8-whitegrid'

plt.figure(figsize=(8,8))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_test, y=y_pred, color = 'dodgerblue', alpha=0.7)
max_val = max(y_test.max(), y_pred.max())
y_pred.max()

plt.plot([0, max_val],[0, max_val], 'r--', lw=2, label='Garis prediksi sempurna (y=x)')
plt.title('Prediksi vs Nilai Aktual', fontsize=16, fontweight='bold')
plt.xlabel('Harga Aktual', fontsize=12)
plt.ylabel('Harga Prediksi', fontsize=12)
plt.legend()
plt.text(0.05,0.9, f'R2:{r2:.4f}', transform=plt.gca(). transAxes, fontsize=12, verticalalignment='top')
plt.show()

# Visualisasi Distribusi Residual Error
plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.histplot(residual, kde=True, color='coral', bins=30)
plt.title('Distribusi Residual Error', fontsize=16, fontweight='bold')
plt.xlabel('Residual (Harga Aktual - Harga Prediksi)', fontsize=12)
plt.axvline(0, color='r', linestyle='--', label='Pusat di Nol')
plt.legend()
plt.show()

# Visualisasi Residual Error vs Nilai Prediksi
plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_pred, y=residual, color='forestgreen', alpha=0.6)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.title('Residual vs Nilai Prediksi', fontsize=16, fontweight='bold')
plt.xlabel('Harga Prediksi', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.show()
