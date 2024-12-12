
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data, target_column):
    """
    Preprocess the dataset by handling missing values and splitting features and target.
    Args:
        data (pd.DataFrame): Input dataset.
        target_column (str): Name of the target column.
    Returns:
        tuple: Processed features (X) and target (y).
    """
    # Drop rows with missing values
    data = data.dropna()

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the data to include in the test split.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Example usage
    from load_data import load_data

    file_path = "baseball.csv"
    target_column = "Playoffs"

    # Load the data
    data = load_data(file_path)

    # Preprocess the data
    X, y = preprocess_data(data, target_column)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
