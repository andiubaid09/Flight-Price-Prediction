from sklearn.model_selection import GridSearchCV

params_grid_XGB = {
    'regressor__regressor__n_estimators' : [150, 200, 300],
    'regressor__regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__regressor__max_depth' : [10, 15, 20],
}

gridsearch_XGB = GridSearchCV(
    estimator= XGB_pipeline,
    param_grid = params_grid_XGB,
    cv = 5,
    scoring = 'r2',
    n_jobs=-1,
    verbose=2
)
gridsearch_XGB.fit(X_train, y_train)
best_XGB = gridsearch_XGB.best_estimator_