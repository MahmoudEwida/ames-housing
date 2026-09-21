import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

def train_regularized_models(X_train, y_train):
    """
    Fit RidgeCV, LassoCV, and ElasticNetCV on training data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        dict: Dictionary containing fitted models: {'ridge': ridge_cv_model, 'lasso': lasso_cv_model, 'enet': enet_cv_model}.
    """
    ridge = RidgeCV().fit(X_train,y_train)
    lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    enet = ElasticNetCV(cv=5, random_state=42).fit(X_train, y_train)
    
    return {
        'ridge': ridge,
        'lasso': lasso,
        'enet': enet
    }
