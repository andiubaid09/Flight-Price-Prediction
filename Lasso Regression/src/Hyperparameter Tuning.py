from sklearn.model_selection import GridSearchCV

grid_lasso = GridSearchCV(
    estimator=pipeline_lasso,
    param_grid=params_grid,
    cv = 5,
    scoring= 'r2',
    n_jobs=-1,
    verbose=1
)
grid_lasso.fit(X_train, y_train)
best_model = grid_lasso.best_estimator_