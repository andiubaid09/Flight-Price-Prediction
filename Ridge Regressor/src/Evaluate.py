from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as plt
import seaborn as sns

print("\n Best alpha:", grid_ridge.best_params_)
print("Best R2 Score (CV) :", grid_ridge.best_score_)
