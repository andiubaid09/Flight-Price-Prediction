from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

regressor = LinearRegression()

model_linear = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor = regressor,
        func = np.log1p,
        inverse_func = np.expm1
    ))
])
model_linear.fit(X_train, y_train)
print(model_linear)