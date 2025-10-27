from sklearn.model_selection import GridSearchCV

params_grid = {
    'regressor__regressor__max_depth': [None, 5, 10, 15, 20, 30, 50],
    'regressor__regressor__min_samples_split' : [2, 5, 10],
    'regressor__regressor__min_samples_leaf' : [1, 2, 5, 10]
}

gridSearch_dt = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=params_grid,
    cv = 5,
    scoring = 'r2',
    n_jobs=-1,
    verbose =2
)

gridSearch_dt.fit(X_train,y_train)
best_dt = gridSearch_dt.best_estimator_