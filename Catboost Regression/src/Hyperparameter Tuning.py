from sklearn.model_selection import RandomizedSearchCV

params_dist = {
    'regressor__regressor__iterations' : [1500, 2000],
    'regressor__regressor__learning_rate' : [0.15, 0.2],
    'regressor__regressor__max_depth' : [10, 15, 20],
    'regressor__regressor__l2_leaf_reg': [7, 10, 15]

}

random_cbm = RandomizedSearchCV(
    estimator= cbm_pipeline,
    param_distributions= params_dist,
    cv = 2,
    n_iter =12,
    n_jobs = -1,
    random_state=42,
    scoring='r2',
    verbose=1
)

