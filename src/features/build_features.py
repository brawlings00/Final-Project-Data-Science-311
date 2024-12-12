# Creating a build_features.py script for feature engineering tasks based on the notebook context.
build_features_script = """
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(data, columns):
    \"\"\"
    Scale numeric features using StandardScaler.
    Args:
        data (pd.DataFrame): Input data.
        columns (list): List of column names to scale.
    Returns:
        pd.DataFrame: Data with scaled numeric features.
    \"\"\"
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def add_interaction_terms(data, columns):
    \"\"\"
    Create interaction terms for specified numeric features.
    Args:
        data (pd.DataFrame): Input data.
        columns (list): List of column names for interaction.
    Returns:
        pd.DataFrame: Data with interaction terms added.
    \"\"\"
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            interaction_col_name = f"{col1}_x_{col2}"
            data[interaction_col_name] = data[col1] * data[col2]
    return data

def encode_categorical_features(data, columns):
    \"\"\"
    One-hot encode categorical features.
    Args:
        data (pd.DataFrame): Input data.
        columns (list): List of categorical columns to encode.
    Returns:
        pd.DataFrame: Data with one-hot encoded categorical features.
    \"\"\"
    return pd.get_dummies(data, columns=columns, drop_first=True)

if __name__ == "__main__":
    # Example usage
    from load_data import load_data

    file_path = "baseball.csv"
    data = load_data(file_path)

    # Example: Scale numeric features
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    data = scale_numeric_features(data, numeric_columns)

    # Example: Add interaction terms
    data = add_interaction_terms(data, numeric_columns[:3])  # Using the first three columns as an example

    # Example: Encode categorical features
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    data = encode_categorical_features(data, categorical_columns)

    print("Processed data preview:")
    print(data.head())
"""

# Saving the script as a Python file
features_output_path = '/mnt/data/build_features.py'
with open(features_output_path, 'w', encoding='utf-8') as f:
    f.write(build_features_script)

features_output_path

