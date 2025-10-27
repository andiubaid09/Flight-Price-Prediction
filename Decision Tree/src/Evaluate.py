from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as plt
import seaborn

print('BEST params yang ditemukan adalah : ', gridSearch_dt.best_params_)
print('BEST CV Score yang ditemukan adalah :', gridSearch_dt.best_score_)
print(best_dt)
y_pred = best_dt.predict(X_test)

print('=== Hasil Evaluasi Model ===')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Hasil rmse model Decision Tree Regressor adalah {rmse:.4f}')
print(f'Hasil MAE model Decision Tree Regressor adalah {mae:.4f}')
print(f'Hasil r2 score Decision Tree Regressor adalah {r2:.4f}')

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