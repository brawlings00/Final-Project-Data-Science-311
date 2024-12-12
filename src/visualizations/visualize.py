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

def plot_feature_importances(model, feature_names):
    """
    Plot the feature importances of a trained model.
    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    sorted_indices = importances.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[sorted_indices], align="center")
    plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_indices])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importances")
    plt.show()

if __name__ == "__main__":
    from load_data import load_data
    from process import preprocess_data, split_data
    from train_model import train_model

    # Load the dataset
    file_path = "baseball.csv"
    target_column = "Playoffs"
    data = load_data(file_path)

    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Plot feature distributions
    plot_feature_distributions(data, numeric_columns, target_column)

    # Plot correlation matrix
    plot_correlation_matrix(data)

    # Train model for feature importance
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)

    # Plot feature importances
    plot_feature_importances(model, X.columns.tolist())
