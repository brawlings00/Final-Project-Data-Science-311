import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

def train_model(X_train, y_train, model_params=None):
    """
    Train a RandomForestClassifier model.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        model_params (dict, optional): Parameters for the RandomForest model.
    Returns:
        RandomForestClassifier: Trained model.
    """
    if model_params is None:
        model_params = {"n_estimators": 100, "random_state": 42}
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.
    Returns:
        dict: Evaluation metrics including accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {"accuracy": accuracy, "report": report}

def save_model(model, file_path):
    """
    Save the trained model to a file.
    Args:
        model: Trained model.
        file_path (str): Path to save the model.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

if __name__ == "__main__":
    from load_data import load_data
    from process import preprocess_data

    # Load the dataset
    file_path = "baseball.csv"
    target_column = "Playoffs"
    data = load_data(file_path)

    # Preprocess the data
    data = data.dropna()  # Handling missing values
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluation = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {evaluation['accuracy']}")
    print("Classification report:")
    print(evaluation["report"])

    # Save the trained model
    save_model(model, "random_forest_model.pkl")

