from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

model_ridge = Ridge()
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', TransformedTargetRegressor(
        regressor= model_ridge,
        func=np.log1p,
        inverse_func=np.expm1
    ))
])
params_grid = {
    'regressor__regressor__alpha' : [0.001, 0.01, 0.1, 1, 10, 100]
}
grid_ridge = GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv = 5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_ridge.fit(X_train, y_train)
best_model = grid_ridge.best_estimator_