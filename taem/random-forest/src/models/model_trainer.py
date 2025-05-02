import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(df: pd.DataFrame, target_column: str, model_save_path: str) -> None:
    """Train a RandomForest model and save it."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # 테스트용 샘플 데이터
    sample_data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 3, 4, 5, 6],
        "price": [100, 150, 200, 250, 300]
    }
    df = pd.DataFrame(sample_data)
    train_model(df, target_column="price", model_save_path="models/random_forest.pkl")
