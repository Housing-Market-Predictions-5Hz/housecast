import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing data."""
    df = df.copy()
    # 예시: 방 개수 대비 가격
    if 'price' in df.columns and 'rooms' in df.columns:
        df['price_per_room'] = df['price'] / df['rooms']
    return df

if __name__ == "__main__":
    # 테스트용 샘플
    sample_data = {
        "price": [300000, 450000, 500000],
        "rooms": [3, 5, 4]
    }
    df = pd.DataFrame(sample_data)
    df = create_features(df)
    print(df.head())
