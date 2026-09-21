# Ames Housing Dataset - Regression Pipeline

Modular scikit-learn pipeline for Ames Housing price prediction.

## Modules

- `step1_load_ames_housing.py`: Load data from OpenML.
- `step2_prepare_target.py`: Extract target and apply `log1p`.
- `step3_handle_feature_types.py`: Numeric, ordinal, and categorical preprocessing with ColumnTransformer.
- `step4_detect_outliers_cooks.py`: Filter extreme outliers.
- `step5_train_regularized_models.py`: Fit RidgeCV, LassoCV, and ElasticNetCV.
