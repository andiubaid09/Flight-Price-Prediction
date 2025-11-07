import matplotlib as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('Best Params ditemukan dengan RandomizedSearchCV :', random_search.best_params_)
print('Best CV Score ditemuakan dengan RandomizedSearchCV :', random_search.best_score_)
print(best_svr)
y_pred = best_svr.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
MAE = mean_absolute_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("=== Hasil Evaluasi Model ===")
print(f"Score R2 yang ditemukan :{R2:.4f}")
print(f"Score MAE yang ditemukan :{MAE:.4f}")
print(f"Score RMSE yang ditemukan {RMSE:.4f}")
