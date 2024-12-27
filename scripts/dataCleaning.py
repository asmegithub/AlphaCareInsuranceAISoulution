import pandas as pd

# Function to drop unnecessary or empty columns
def drop_unnecessary_columns(df, columns_to_drop):
    """
    Drops unnecessary or fully empty columns.
    """
    return df.drop(columns=columns_to_drop)

# Function to handle missing values for numerical columns
def fill_missing_numerical(df, numerical_columns, strategy='median'):
    """
    Fill missing values in numerical columns with a given strategy (mean/median).
    """
    for column in numerical_columns:
        if strategy == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
    return df

# Function to handle missing values for categorical columns
def fill_missing_categorical(df, categorical_columns):
    """
    Fill missing values in categorical columns with the most frequent value (mode).
    """
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df

# Function to convert columns to datetime format
def convert_to_datetime(df, datetime_columns):
    """
    Convert specified columns to datetime format.
    """
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column])
    return df


