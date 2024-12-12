
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def check_missing_values(data):
    """
    Check for missing values in the dataset.
    Args:
        data (pd.DataFrame): Input data.
    Returns:
        pd.Series: Count of missing values per column.
    """
    return data.isnull().sum()

def visualize_distribution(data, column):
    """
    Visualize the distribution of a categorical column.
    Args:
        data (pd.DataFrame): Input data.
        column (str): Name of the column to visualize.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=column, data=data)
    plt.title(f'Distribution of {column}')
    plt.show()

def compute_correlations(data, target_column):
    """
    Compute and visualize correlations of numeric features with the target column.
    Args:
        data (pd.DataFrame): Input data.
        target_column (str): Name of the target column.
    """
    numeric_features = data.select_dtypes(include=['int64', 'float64']).drop(columns=[target_column])
    correlations = numeric_features.corrwith(data[target_column])
    
    plt.figure(figsize=(8, 6))
    correlations.sort_values().plot(kind='barh', color='skyblue')
    plt.title(f'Correlation of Features with {target_column}')
    plt.xlabel('Correlation Coefficient')
    plt.show()
    return correlations

if __name__ == "__main__":
    # Example usage
    file_path = "baseball.csv"
    target_column = "Playoffs"

    # Load the data
    data = load_data(file_path)

    # Check for missing values
    missing_values = check_missing_values(data)
    print("Missing values per column:")
    print(missing_values)

    # Visualize the distribution of the target column
    visualize_distribution(data, target_column)

    # Compute and visualize correlations with the target column
    correlations = compute_correlations(data, target_column)
    print("Correlations with target column:")
    print(correlations)
