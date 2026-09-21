import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def numeric_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"), 
        StandardScaler()
    )
def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    
def ordinal_pipeline(num_cols):
    # qual_order repeated for each ordinal column
    qual_order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Po"),
        OrdinalEncoder(categories=[qual_order] * num_cols)
    )
    
def handle_feature_types(df):
    """
    Handle missing values, ordinal mappings (e.g. ExterQual), and categorical encodings.

    Args:
        df (pd.DataFrame): Feature dataframe X.

    Returns:
        pd.DataFrame: Processed dataframe with handled missing values and encodings.
    """
    potential_ordinal_cols = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']
    ordinal_cols = [col for col in potential_ordinal_cols if col in df.columns]

    # 2. Find numeric columns (int, float)
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'number']).columns.tolist()

    # 3. Find remaining categorical columns (text/object, but NOT the ordinal ones)
    categorical_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns 
        if col not in ordinal_cols
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline(),numeric_cols),
            ('ord',ordinal_pipeline(len(ordinal_cols)),ordinal_cols),
            ('cat',categorical_pipeline(),categorical_cols)
        ]
    )
    
    preprocessor.set_output(transform='pandas')
    
    df_preprocessed = preprocessor.fit_transform(df)
    
    return df_preprocessed
    
