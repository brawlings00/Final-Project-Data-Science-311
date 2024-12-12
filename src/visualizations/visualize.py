import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(data, numeric_columns, target_column):
    """
    Plot the distribution of numeric features, separated by the target variable.
    Args:
        data (pd.DataFrame): Dataset.
        numeric_columns (list): List of numeric column names.
        target_column (str): Name of the target column.
    """
    for column in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data, x=column, hue=target_column, kde=True, bins=30)
        plt.title(f"Distribution of {column} by {target_column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.legend(title=target_column)
        plt.show()

def plot_correlation_matrix(data):
    """
    Plot the correlation matrix of numeric features in the dataset.
    Args:
        data (pd.DataFrame): Dataset.
    """
    plt.figure(figsize=(10, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_feature
