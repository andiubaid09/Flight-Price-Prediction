from sklearn.model_selection import RandomizedSearchCV

params_dist = {
    'regressor__regressor__C': np.logspace(0, 2.5, 8),                  # dari 1 sampai 300
    'regressor__regressor__epsilon': np.linspace(0.05, 0.3, 5),        # margin toleransi
    'regressor__regressor__gamma': ['scale', 'auto']                    # Hanya digunakan oleh kernel non-linear
}

random_searchSVR = RandomizedSearchCV(
    estimator = svr_pipeline,
    param_distributions=params_dist,
    n_iter = 3,
    scoring = 'r2',
    cv = 2,
    verbose = 2,
    n_jobs = -1,
    random_state=42
)

random_searchSVR.fit(X_train, y_train)
best_svr = random_search.best_estimator_