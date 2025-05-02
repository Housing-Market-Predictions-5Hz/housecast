import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path to import the RandomForestModel class
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.random_forest_model import RandomForestModel

def load_data(data_path):
    """
    Load and preprocess data.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file
        
    Returns:
    --------
    X : DataFrame
        Features
    y : Series
        Target variable
    """
    # For this example, we'll assume the data is in CSV format
    # Replace this with your actual data loading logic
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        
        # Basic preprocessing
        # Drop rows with missing values
        df = df.dropna()
        
        # For demonstration, we'll assume the last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create sample data as fallback
        print("Creating sample data instead...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, 
                                  n_informative=10, n_redundant=5,
                                  random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y, name='target')
        return X, y

def train_and_evaluate_model(X, y, model_type='classifier', output_dir=None):
    """
    Train and evaluate a random forest model.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    model_type : str, default='classifier'
        Type of model to train ('classifier' or 'regressor')
    output_dir : str, default=None
        Directory to save model and plots
        
    Returns:
    --------
    model : RandomForestModel
        Trained model
    metrics : dict
        Evaluation metrics
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Feature preprocessing
    # For demonstration, we'll just standardize the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    # Initialize the model
    model = RandomForestModel(
        model_type=model_type,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Train the model
    X_train, X_test, y_train, y_test = model.train(X_scaled, y, test_size=0.3)
    print(f"Model trained on {X_train.shape[0]} samples")
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):
            print(f"{metric}:")
            print(value)
        else:
            print(f"{metric}: {value:.4f}")
    
    # Plot and save feature importance
    fig = model.plot_feature_importance(top_n=15)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    else:
        plt.show()
    
    # Save the model
    if output_dir:
        model_path = os.path.join(output_dir, 'random_forest_model.joblib')
        model.save_model(model_path)
    
    return model, metrics

def hyperparameter_tuning(X, y, model_type='classifier'):
    """
    Perform hyperparameter tuning for the random forest model.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    model_type : str, default='classifier'
        Type of model to train ('classifier' or 'regressor')
        
    Returns:
    --------
    best_model : RandomForestModel
        Best model after tuning
    tuning_results : dict
        Results of hyperparameter tuning
    """
    # Initialize base model
    model = RandomForestModel(model_type=model_type)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Starting hyperparameter tuning (this may take a while)...")
    # Perform hyperparameter tuning
    tuning_results = model.hyperparameter_tuning(X, y, param_grid, cv=5)
    
    print("\nHyperparameter Tuning Results:")
    print(f"Best parameters: {tuning_results['best_params']}")
    print(f"Best cross-validation score: {tuning_results['best_score']:.4f}")
    
    return model, tuning_results

def main():
    """Main function to execute the model training pipeline."""
    # Define paths
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'models')
    
    # For demonstration purposes, use a sample data path
    # Replace with your actual data path
    data_path = os.path.join(data_dir, 'processed_data.csv')
    
    print("=== Random Forest Model Training Pipeline ===")
    
    # Load data
    X, y = load_data(data_path)
    
    # Check if it's a classification or regression problem
    if len(np.unique(y)) <= 10:  # Arbitrary threshold, adjust as needed
        model_type = 'classifier'
        print("Detected a classification problem")
    else:
        model_type = 'regressor'
        print("Detected a regression problem")
    
    # Train and evaluate model
    model, metrics = train_and_evaluate_model(X, y, model_type=model_type, output_dir=output_dir)
    
    # Uncomment to run hyperparameter tuning (time-consuming)
    # best_model, tuning_results = hyperparameter_tuning(X, y, model_type=model_type)
    # best_model.save_model(os.path.join(output_dir, 'best_random_forest_model.joblib'))
    
    print("\n=== Model Training Complete ===")
    
if __name__ == "__main__":
    main() 