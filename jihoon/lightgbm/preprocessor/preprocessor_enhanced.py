import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from preprocessor.column_tags import TAGS

CURRENT_YEAR = 2023
GANGNAM_CENTER_X = 203731
GANGNAM_CENTER_Y = 452331
EARTH_RADIUS = 6371  # km

def apply_target_encoding(train_df, test_df, col_name, target_name="target"):
    target_map = train_df.groupby(col_name)[target_name].mean()
    global_mean = train_df[target_name].mean()
    train_encoded = train_df[col_name].map(target_map).fillna(global_mean)
    test_encoded = test_df[col_name].map(target_map).fillna(global_mean)
    new_col = f"{col_name}_te"
    train_df[new_col] = train_encoded
    test_df[new_col] = test_encoded
    return train_df, test_df

def coord_cols(df):
    # 위도, 경도 순서로 변환
    return np.deg2rad(df[['좌표Y', '좌표X']].values)

def build_ball_tree(df):
    coords = coord_cols(df)
    return BallTree(coords, metric='haversine')

def add_transport_features(df, bus_df, subway_df, radius_km):
    for df_temp in [bus_df, subway_df]:
        df_temp.rename(columns={"X좌표": "좌표X", "Y좌표": "좌표Y"}, inplace=True)

    if "좌표Y" not in df.columns or "좌표X" not in df.columns:
        df["num_subway_400m"] = 0
        df["num_bus_400m"] = 0
        df["교통_총밀도_400m"] = 0
        return df

    df["num_subway_400m"] = 0
    df["num_bus_400m"] = 0

    valid_idx = df["좌표Y"].notna() & df["좌표X"].notna()
    df_coords = df.loc[valid_idx].copy()
    query_coords = coord_cols(df_coords)

    subway_tree = build_ball_tree(subway_df)
    bus_tree = build_ball_tree(bus_df)
    radius_radian = 0.4 / EARTH_RADIUS

    subway_counts = subway_tree.query_radius(query_coords, r=radius_radian, count_only=True)
    bus_counts = bus_tree.query_radius(query_coords, r=radius_radian, count_only=True)

    df.loc[valid_idx, "num_subway_400m"] = subway_counts
    df.loc[valid_idx, "num_bus_400m"] = bus_counts

    df["num_subway_400m"] = df["num_subway_400m"].fillna(0).astype(int)
    df["num_bus_400m"] = df["num_bus_400m"].fillna(0).astype(int)
    df["교통_총밀도_400m"] = df["num_subway_400m"] + df["num_bus_400m"]
    return df

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

def add_leading_apt_flags(df, distance_col="대장아파트거리"):
    df["대장_근접여부_100m"] = (df[distance_col] <= 100).astype(int)

def load_data(train_path, test_path, bus_path, subway_path, submission_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    bus = pd.read_csv(bus_path)
    subway = pd.read_csv(subway_path)
    submission = pd.read_csv(submission_path)
    return train, test, bus, subway, submission

def preprocess_enhanced(train, test, bus, subway, radius_km=0.4):
    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = np.nan
    combined = pd.concat([train, test], axis=0)

    if {'계약년월', '계약일'}.issubset(combined.columns):
        combined['계약일자_dt'] = pd.to_datetime(
            combined['계약년월'].astype(str) + combined['계약일'].astype(str).str.zfill(2),
            format="%Y%m%d", errors='coerce'
        )
        combined['계약_월'] = combined['계약일자_dt'].dt.month.fillna(0).astype(int)
        combined['계약_계절'] = (combined['계약일자_dt'].dt.month % 12 // 3 + 1).fillna(0).astype(int)
        combined['계약_연'] = combined['계약일자_dt'].dt.year.fillna(0).astype(int)
        combined['계약_일자'] = combined['계약일자_dt'].dt.day.fillna(0).astype(int)
        combined = combined.drop(columns=['계약일자_dt'])

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
        combined["층수_면적_비율"] = combined["층"] / (combined["전용면적(㎡)"] + 1e-3)

    if {"좌표X", "좌표Y"}.issubset(combined.columns):
        combined["distance_from_gangnam_center"] = np.sqrt(
            (combined["좌표X"] - GANGNAM_CENTER_X) ** 2 + (combined["좌표Y"] - GANGNAM_CENTER_Y) ** 2
        )

    combined = add_transport_features(combined, bus, subway, radius_km)

    if {"시군구", "아파트명"}.issubset(combined.columns):
        combined["시군구_아파트명"] = combined["시군구"].astype(str) + "_" + combined["아파트명"].astype(str)

    if "대장아파트" in combined.columns:
        combined["대장아파트"] = combined["대장아파트"].fillna("미상").astype("category")
        combined["is_대장단지"] = (combined["아파트명"] == combined["대장아파트"]).astype(int)
        combined["대장아파트"] = combined["대장아파트"].cat.codes

    if "대장아파트거리" in combined.columns:
        combined["대장아파트거리"] = combined["대장아파트거리"].fillna(combined["대장아파트거리"].median())
        combined["대장_거리_log"] = np.log1p(combined["대장아파트거리"])
        add_leading_apt_flags(combined)

    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])

    for col in ["전용면적_bin", "시군구_아파트명"]:
        if col in train_processed.columns:
            train_processed, test_processed = apply_target_encoding(train_processed, test_processed, col)
            train_processed.drop(columns=col, inplace=True)
            test_processed.drop(columns=col, inplace=True)

    return train_processed, test_processed, bus, subway