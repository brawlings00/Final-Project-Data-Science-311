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
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolis
