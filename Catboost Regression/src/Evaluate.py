import matplotlib as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('Best Params yang ditemukan', random_cbm.best_params_)
print('CV Score yang ditemukan', random_cbm.best_score_)
print(best_cbm)
y_pred = best_cbm.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("=== Hasil Evaluasi Model ===")
print(f"Score R2 yang ditemukan :{R2:.4f}")
print(f"Score MAE yang ditemukan :{MAE:.4f}")
print(f"Score RMSE yang ditemukan {RMSE:.4f}")

# Visualisasi Prediksi vs Nilai Aktual
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
plt.text(0.05,0.9, f'R2:{R2:.4f}', transform=plt.gca(). transAxes, fontsize=12, verticalalignment='top')
plt.show()

# Visualisasi Distribusi Residual Error