import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

# Add parent directory to path to import the RandomForestModel class
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.random_forest_model import RandomForestModel

def load_test_data(data_path):
    """
    Load and preprocess test data.
    
    Parameters:
    -----------
    data_path : str
        Path to the test data file
        
    Returns:
    --------
    X_test : DataFrame
        Features for testing
    y_test : Series or None
        Target variable for testing (if available)
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        print(f"Test data loaded successfully with shape: {df.shape}")
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Check if target column exists
        if 'target' in df.columns:
            X_test = df.drop('target', axis=1)
            y_test = df['target']
            print(f"Features shape: {X_test.shape}, Target shape: {y_test.shape}")
            return X_test, y_test
        else:
            # Assume all columns are features
            X_test = df
            print(f"Features shape: {X_test.shape}, No target column found.")
            return X_test, None
    
    except Exception as e:
        print(f"Error loading test data: {e}")
        # Create sample test data as fallback
        print("Creating sample test data instead...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=20, 
                                  n_informative=10, n_redundant=5,
                                  random_state=43)  # Different random state from training
        X_test = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_test = pd.Series(y, name='target')
        return X_test, y_test

def load_scaler(scaler_path):
    """
    Load the feature scaler.
    
    Parameters:
    -----------
    scaler_path : str
        Path to the saved scaler object
        
    Returns:
    --------
    scaler : StandardScaler
        Loaded scaler object or None if not found
    """
    try:
        scaler = joblib.load(scaler_path)
        print("Feature scaler loaded successfully")
        return scaler
    except:
        print("Feature scaler not found, will use default standardization")
        return None

def make_predictions(model, X_test, scaler=None):
    """
    Make predictions on test data.
    
    Parameters:
    -----------
    model : RandomForestModel
        Trained model
    X_test : DataFrame
        Features for testing
    scaler : StandardScaler, default=None
        Scaler for feature normalization
        
    Returns:
    --------
    y_pred : array
        Predicted values
    y_proba : array or None
        Predicted probabilities (only for classification)
    """
    # Preprocess test features
    if scaler:
        X_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
    else:
        # Apply standard scaling if scaler not provided
        temp_scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            temp_scaler.fit_transform(X_test),
            columns=X_test.columns
        )
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Try to get probability estimates (only for classification)
    try:
        y_proba = model.predict_proba(X_scaled)
        return y_pred, y_proba
    except:
        return y_pred, None

def evaluate_predictions(y_test, y_pred, y_proba=None, model_type='classifier'):
    """
    Evaluate prediction performance.
    
    Parameters:
    -----------
    y_test : Series
        True target values
    y_pred : array
        Predicted values
    y_proba : array, default=None
        Predicted probabilities (for classification)
    model_type : str, default='classifier'
        Type of model ('classifier' or 'regressor')
        
    Returns:
    --------
    None
    """
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    if y_test is None:
        print("No ground truth available for evaluation.")
        return
    
    if model_type == 'classifier':
        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # ROC curve for binary classification
        if y_proba is not None and len(np.unique(y_test)) == 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
    else:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nRegression Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.show()
        
        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

def save_predictions(y_pred, output_path, X_test=None):
    """
    Save predictions to a CSV file.
    
    Parameters:
    -----------
    y_pred : array
        Predicted values
    output_path : str
        Path to save predictions
    X_test : DataFrame, default=None
        Test features (optional to include in output)
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create DataFrame with predictions
    if X_test is not None:
        # Include features with predictions
        result_df = X_test.copy()
        result_df['prediction'] = y_pred
    else:
        # Only predictions
        result_df = pd.DataFrame({'prediction': y_pred})
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    """Main function to execute the prediction pipeline."""
    # Define paths
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data', 'test')
    results_dir = os.path.join(base_dir, 'reports', 'predictions')
    
    # Model and data paths (adjust as needed)
    model_path = os.path.join(models_dir, 'random_forest_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')  # Optional
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    output_path = os.path.join(results_dir, 'predictions.csv')
    
    print("=== Random Forest Model Prediction Pipeline ===")
    
    # Load model
    try:
        model = RandomForestModel.load_model(model_path)
        model_type = model.model_type
        print(f"Loaded {model_type} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Cannot proceed without a model.")
        return
    
    # Load scaler (optional)
    scaler = load_scaler(scaler_path)
    
    # Load test data
    X_test, y_test = load_test_data(test_data_path)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred, y_proba = make_predictions(model, X_test, scaler)
    
    # Evaluate predictions (if ground truth is available)
    if y_test is not None:
        evaluate_predictions(y_test, y_pred, y_proba, model_type)
    
    # Save predictions
    save_predictions(y_pred, output_path, X_test)
    
    print("\n=== Prediction Complete ===")

if __name__ == "__main__":
    main() 