from sklearn.model_selection import GridSearchCV

# Pembagunan Pipeline akhir
# Perhatian : Gunakan format "regressor__regressor__parameter_name"
# Di dalam TransformedTargetResessor ada properti regressor yang menyimpan model rfr

param_grid = {
    # 1 Jumlah pohon
    'regressor__regressor__n_estimators':[100,200,],
    # 2 Kedalaman pohon
    'regressor__regressor__max_depth':[5,15],
    # 3 Jumlah sampel minimum untuk split
    'regressor__regressor__min_samples_split':[2 , 5]
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(
    estimator=rfr_pipeline,
    param_grid=param_grid,
    cv=3, #Cross validation
    scoring='neg_mean_squared_error',
    n_jobs=2,
    verbose=1

)

# Melakukan pencarian GridSearchCV
print("Memulai pencarian Grid... Proses ini akan melatih pipeline berulang kali")
grid_search.fit(X_train, y_train)
print("Pencarian Grid Selesai")

# best_estimators adalah pipeline final yang sudah di tune
best_model = grid_search.best_estimator_
print("\n ==== Hasil Tuning =====")
print(f"Parameter terbaik : {grid_search.best_params_}")
print(f"Skor terbaik CV (Neg.MSE):{grid_search.best_score_:.4f}")