from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib as plt
import seaborn as sns

print('Best Params yang ditemukan adalah :', grid_elastic.best_params_)
print('Best CV Scoring yang ditemukan adalah :', grid_elastic.best_score_)
print(best_elastic)
y_pred = best_elastic.predict(X_test)
