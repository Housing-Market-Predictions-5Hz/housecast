import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from a given filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")
    return pd.read_csv(filepath)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to a CSV file at the given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    # Example usage
    try:
        df = load_data('data/raw/sample.csv')
        print(df.head())
    except FileNotFoundError as e:
        print(e)
