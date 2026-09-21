import numpy as np
import pandas as pd

def prepare_target(df):
    """
    Extract target 'SalePrice' and apply log1p transformation.

    Args:
        df (pd.DataFrame): Raw housing dataframe.

    Returns:
        tuple: (X, y) where X is a pd.DataFrame of features (without 'SalePrice' or 'Id')
               and y is a pd.Series of log1p transformed 'SalePrice'.
    """
    y = np.log1p(df['SalePrice'])
    X = df.drop(columns = ['SalePrice','Id'], errors = 'ignore')
    return X,y
