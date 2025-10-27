from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as plt
import seaborn

print('BEST params yang ditemukan adalah : ', gridSearch_dt.best_params_)
print('BEST CV Score yang ditemukan adalah :', gridSearch_dt.best_score_)
print(best_dt)
y_pred = best_dt.predict(X_test)