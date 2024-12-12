import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the performance of a classification model.
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test set features.
        y_test (pd.Series): Test set true labels.
    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, F1 score).
    """
    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary"),
        "recall": recall_score(y_test, y_pred, average="binary"),
        "f1_score": f1_score(y_test, y_pred, average="binary"),
    }

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, class_names):
    """
    Plot a confusion matrix to visualize model performance.
    Args:
        y_test (pd.Series): Test set true labels.
        y_pred (pd.Series): Predicted labels.
        class_names (list): List of class names for the matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    from load_data import load_data
    from process import preprocess_data, split_data
    from train_model import train_model

    # Load the data
    file_path = "baseball.csv"
    target_column = "Playoffs"
    data = load_data(file_path)

    # Preprocess and split the data
    X, y = preprocess_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train a model
    model = train_model(X_train, y_train)

    # Evaluate the model
    metrics, y_pred = evaluate_classification_model(model, X_test, y_test)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot the confusion matrix
    class_names = ["No Playoffs", "Playoffs"]
    plot_confusion_matrix(y_test, y_pred, class_names)
