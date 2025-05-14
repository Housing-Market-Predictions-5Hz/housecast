import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from preprocessor.column_tags import TAGS

CURRENT_YEAR = 2023
GANGNAM_CENTER_X = 203731
GANGNAM_CENTER_Y = 452331
EARTH_RADIUS = 6371  # km

# 위경도 -> 라디안
def coord_cols(df):
    return np.deg2rad(df[['lat', 'lng']].values)

def build_ball_tree(df):
    coords = coord_cols(df)
    return BallTree(coords, metric='haversine')

def add_transport_features(df, bus_df, subway_df, radius_km):
    if "lat" not in df.columns or "lng" not in df.columns:
        df["num_subway_1km"] = 0
        df["num_bus_1km"] = 0
        return df

    df_coords = df.dropna(subset=['lat', 'lng']).copy()
    query_coords = coord_cols(df_coords)
    subway_tree = build_ball_tree(subway_df)
    bus_tree = build_ball_tree(bus_df)
    radius_radian = radius_km / EARTH_RADIUS
    subway_counts = subway_tree.query_radius(query_coords, r=radius_radian, count_only=True)
    bus_counts = bus_tree.query_radius(query_coords, r=radius_radian, count_only=True)
    df_coords["num_subway_1km"] = subway_counts
    df_coords["num_bus_1km"] = bus_counts

    df = df.merge(df_coords[["num_subway_1km", "num_bus_1km"]], left_index=True, right_index=True, how="left")
    df[["num_subway_1km", "num_bus_1km"]] = df[["num_subway_1km", "num_bus_1km"]].fillna(0)
    return df

def load_complex_code(path="data/Complex Code_20221129.xlsx"):
    df = pd.read_excel(path)
    df["매칭키"] = df["시군구"].astype(str) + "_" + df["건물명"].astype(str)
    return df[["매칭키", "단지코드"]].dropna()

def bin_area(x):
    if x < 60: return '소형'
    elif x < 85: return '중형'
    elif x < 135: return '대형'
    else: return '초대형'

def bin_floor(x):
    if pd.isna(x): return '미상'
    if x <= 5: return '저층'
    elif x <= 15: return '중층'
    else: return '고층'

def load_data(train_path, test_path, bus_path, subway_path, submission_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    bus = pd.read_csv(bus_path)
    subway = pd.read_csv(subway_path)
    submission = pd.read_csv(submission_path)
    return train, test, bus, subway, submission

def preprocess_enhanced(train, test, bus, subway, radius_km=1.0):
    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = np.nan
    combined = pd.concat([train, test], axis=0)

    complex_map = load_complex_code()
    combined["매칭키"] = combined["시군구"].astype(str) + "_" + combined["아파트명"].astype(str)
    combined = pd.merge(combined, complex_map, how="left", on="매칭키")

    if "단지코드" not in combined.columns:
        raise KeyError("❌ '단지코드' 컬럼 누락 → 외부 매핑 확인 필요")

    for col, tag in TAGS.items():
        if col not in combined.columns:
            continue
        if tag == "drop":
            combined = combined.drop(columns=col)
        elif tag == "flag":
            combined[f"is_na_{col}"] = combined[col].isnull().astype(int)
            combined = combined.drop(columns=col)
        elif tag == "impute":
            if combined[col].dtype == "object":
                combined[col] = combined[col].fillna("미상").astype("category").cat.codes
            else:
                combined[col] = combined[col].fillna(combined[col].median())
        elif tag == "coord":
            combined[col] = combined[col].fillna(0)
        elif tag == "categorical":
            combined[col] = combined[col].fillna("미상").astype("category").cat.codes
        elif tag == "keep":
            if combined[col].dtype == "object":
                combined[col] = combined[col].fillna("미상")
            else:
                combined[col] = combined[col].fillna(combined[col].median())

    if "건축년도" in combined.columns:
        combined["building_age"] = (CURRENT_YEAR - combined["건축년도"]).clip(lower=0)

    if "전용면적(㎡)" in combined.columns:
        combined["전용면적_bin"] = combined["전용면적(㎡)"].apply(bin_area).astype("category").cat.codes

    if "층" in combined.columns:
        combined["층수_bin"] = combined["층"].apply(bin_floor).astype("category").cat.codes
        if "전용면적(㎡)" in combined.columns:
            combined["floor_x_area"] = combined["층"] * combined["전용면적(㎡)"]

    if "좌표X" in combined.columns and "좌표Y" in combined.columns:
        combined["distance_from_gangnam_center"] = np.sqrt(
            (combined["좌표X"] - GANGNAM_CENTER_X) ** 2 + (combined["좌표Y"] - GANGNAM_CENTER_Y) ** 2
        )

    if "시군구" in combined.columns and "단지코드" in combined.columns:
        combined["시군구_단지"] = (combined["시군구"].astype(str) + "_" + combined["단지코드"].astype(str))\
            .astype("category").cat.codes

    if "target" in combined.columns:
        mean_target = combined[combined["is_train"] == 1].groupby("단지코드")["target"].mean()
        combined["단지별_평균가"] = combined["단지코드"].map(mean_target)
        combined["단지별_평균가"] = combined["단지별_평균가"].fillna(combined["단지별_평균가"].median())

    if "lat" in combined.columns and "lng" in combined.columns:
        combined = add_transport_features(combined, bus, subway, radius_km)

    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])

    return train_processed, test_processed, bus, subway
