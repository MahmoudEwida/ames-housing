import pandas as pd

def detect_outliers_cooks(X, y):
    """
    Identify extreme outliers (e.g., GrLivArea > 4000).

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.

    Returns:
        pd.Index: Index labels of extreme outliers to be dropped.
    """
    outlier_mask = X['GrLivArea'] > 4000
    return X.index[outlier_mask]
