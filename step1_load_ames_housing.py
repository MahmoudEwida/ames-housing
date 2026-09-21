from sklearn.datasets import fetch_openml

def load_ames_housing():
    """
    Fetch the Ames Housing dataset from OpenML ('house_prices') as a pandas DataFrame.

    Returns:
        pd.DataFrame: The raw Ames Housing dataframe.
    """
    housing = fetch_openml(name='house_prices', as_frame= True, parser='auto')
    return housing.frame


# print(load_ames_housing())