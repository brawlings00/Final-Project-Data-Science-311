import argparse
from load_data import load_data
from process import preprocess_data, split_data
from build_features import scale_numeric_features, encode_categorical_features, add_interaction_terms
from train_model import train_model, save_model
from evaluate_model import evaluate_classification_model, plot_confusion_matrix
from visualize import plot_feature_distributions, plot_correlation_matrix, plot_feature_importances

def main(file_path, target_column, model_output_path):
    # Step 1: Load the data
    print("Loading data...")
    data = load_data(file_path)

    # Step 2: Visualize the data
    print("Visualizing data...")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
    plot_feature_distributions(data, numeric_columns, target_column)
    plot_correlation_matrix(data)

    # Step 3: Preprocess the data
    print("Preprocessing data...")
    data = data.dropna()  # Handle missing values
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Scale and encode features
    X = scale_numeric_features(X, numeric_columns)
    X = encode_categorical_features(X, categorical_columns)
    X = add_interaction_terms(X, numeric_columns[:3])  # Add interaction terms

    # Step 4: Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train the model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Step 6: Evaluate the model
    print("Evaluating model...")
    metrics, y_pred = evaluate_classification_model(model, X_test, y_test)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot confusion matrix
    class_names = ["No Playoffs", "Playoffs"]
    plot_confusion_matrix(y_test, y_pred, class_names)

    # Step 7: Visualize feature importances
    print("Visualizing feature importances...")
    plot_feature_importances(model, X.columns.tolist())

    # Step 8: Save the model
    print("Saving model...")
    save_model(model, model_output_path)
    print("Workflow completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the machine learning pipeline for the baseball dataset.")
    parser.add_argument("--file_path", type=str, default="baseball.csv", help="Path to the input dataset.")
    parser.add_argument("--target_column", type=str, default="Playoffs", help="Name of the target column.")
    parser.add_argument("--model_output_path", type=str, default="random_forest_model.pkl", help="Path to save the trained model.")
    args = parser.parse_args()

    main(args.file_path, args.target_column, args.model_output_path)

