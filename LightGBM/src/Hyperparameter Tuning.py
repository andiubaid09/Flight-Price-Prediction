from sklearn.model_selection import GridSearchCV

params_grid = {
    'regressor__regressor__n_estimators' : [200, 300, 500],
    'regressor__regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__regressor__max_depth' : [10, 15, 20]
}

gridsearchLBGM = GridSearchCV(
    estimator= lbgm_pipeline,
    param_grid= params_grid,
    cv = 5,
    scoring='r2',
    n_jobs=-1,
    verbose =2
)
gridsearchLBGM.fit(X_train, y_train)
best_LGBM = gridsearchLBGM.best_estimator_
print(type(X_test))
print(X_test.columns)