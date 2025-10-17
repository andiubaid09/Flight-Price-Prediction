from sklearn.model_selection import GridSearchCV

param_grid_elastic = {
    'regressor__regressor__alpha' : [0.001, 0.01, 0.1, 1, 10, 100],
    'regressor__regressor__l1_ratio' : [0.2, 0.5, 0.8]
}
grid_elastic = GridSearchCV(
    pipeline_elastic,
    param_grid=param_grid_elastic,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose =2
)
grid_elastic.fit(X_train, y_train)
best_elastic = grid_elastic.best_estimator_