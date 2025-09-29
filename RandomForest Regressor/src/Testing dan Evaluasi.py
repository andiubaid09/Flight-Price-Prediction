from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Melakukan prediksi
y_pred = best_model.predict(X_test)

# Hasil Evaluasi pada Data Uji
print('\n --- Hasil Evaluasi pada Data Uji ---')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse  = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f'MSE (Mean Absolute Error): {mse:.4f}')
print(f"R-Squared (R2 Score): {r2:.4f}")
print(f'Harga Aktual Data Uji:\n{y_test.values}')
print(f"Harga Prediksi (Skala asli):\n{np.round(y_pred,2)}")

# Visualisasi Hasil Harga Prediksi vs Harga Aktual

seaborn_version = 'seaborn-v0_8-whitegrid'
plt.figure(figsize=(8,8))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_test, y=y_pred, color='dodgerblue', alpha=0.7)
max_val = max(y_test.max(), y_pred.max())
y_pred.max()

plt.plot([0, max_val], [0,max_val], 'r--', lw=2, label='Garis Prediksi Sempurna (y=x)')
plt.title(" Prediksi vs Nilai Aktual", fontsize=16, fontweight='bold')
plt.xlabel('Harga Aktual', fontsize=12)
plt.ylabel('Harga Prediksi', fontsize=12)
plt.legend()
plt.text(0.05, 0.9, f'$R^2$: {r2:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top')
plt.show() 

# Visualisasi Distribusi Residual Error
residual = y_test - y_pred

plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.histplot(residual, kde=True, color='coral', bins=30)
plt.title('Distribusi Residual Error', fontsize=16, fontweight='bold')
plt.xlabel('Residuals (Harga Aktual - Harga Prediksi)', fontsize=12)
plt.axvline(0, color='r', linestyle='--', label='Pusat di Nol')
plt.legend()
plt.show()

# Visualisasi Residual vs Nilai Prediksi
plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)

sns.scatterplot(x=y_pred, y=residual, color='forestgreen', alpha=0.6)
plt.axhline(0, color='r', linestyle='--', lw=2) # Garis residual nol
plt.title('Residual vs Nilai Prediksi', fontsize=16, fontweight='bold')
plt.xlabel('Harga Prediksi', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.show()

# Visualisasi Feature Importance
TransformedTargetRegressor
rfr_best = best_model.named_steps['regressor'].regressor_
importances = rfr_best.feature_importances_

feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance':importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
plt.style.use(seaborn_version)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Fitur Penting (Gini Importance)', fontsize=16, fontweight='bold')
plt.xlabel('Gini Importance', fontsize=12)
plt.ylabel('Fitur', fontsize=12)
plt.tight_layout()
plt.show()
