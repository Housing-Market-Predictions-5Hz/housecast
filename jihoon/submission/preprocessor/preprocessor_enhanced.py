import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from preprocessor.column_tags import TAGS

CURRENT_YEAR = 2023
GANGNAM_CENTER_X = 203731
GANGNAM_CENTER_Y = 452331
EARTH_RADIUS = 6371  # km 지구 반지름 (BallTree 거리 계산용)

# Target Encoding 적용 함수
def apply_target_encoding(train_df, test_df, col_name, target_name="target"):
    target_map = train_df.groupby(col_name)[target_name].mean()
    global_mean = train_df[target_name].mean()
    train_encoded = train_df[col_name].map(target_map).fillna(global_mean)
    test_encoded = test_df[col_name].map(target_map).fillna(global_mean)
    new_col = f"{col_name}_te"
    train_df[new_col] = train_encoded
    test_df[new_col] = test_encoded
    return train_df, test_df

# 위경도를 라디안으로 변환 (BallTree 계산용)
def coord_cols(df):
    return np.deg2rad(df[['좌표Y', '좌표X']].values)

# BallTree 생성 함수 (위경도 기반 거리 계산)
def build_ball_tree(df):
    coords = coord_cols(df)
    return BallTree(coords, metric='haversine')

# 반경 내 교통 밀도 feature 생성 (1km 기준)
def add_transport_features(df, bus_df, subway_df, radius_km):
    for df_temp in [bus_df, subway_df]:
        df_temp.rename(columns={"X좌표": "좌표X", "Y좌표": "좌표Y"}, inplace=True)

    if "좌표Y" not in df.columns or "좌표X" not in df.columns:
        df["num_subway_1km"] = 0
        df["num_bus_1km"] = 0
        df["교통_총밀도_1km"] = 0
        return df

    df["num_subway_1km"] = 0
    df["num_bus_1km"] = 0

    valid_idx = df["좌표Y"].notna() & df["좌표X"].notna()
    query_coords = coord_cols(df.loc[valid_idx])

    subway_tree = build_ball_tree(subway_df)
    bus_tree = build_ball_tree(bus_df)
    radius_radian = radius_km / EARTH_RADIUS

    subway_counts = subway_tree.query_radius(query_coords, r=radius_radian, count_only=True)
    bus_counts = bus_tree.query_radius(query_coords, r=radius_radian, count_only=True)

    df.loc[valid_idx, "num_subway_1km"] = subway_counts
    df.loc[valid_idx, "num_bus_1km"] = bus_counts

    df["num_subway_1km"] = df["num_subway_1km"].fillna(0).astype(int)
    df["num_bus_1km"] = df["num_bus_1km"].fillna(0).astype(int)
    df["교통_총밀도_1km"] = df["num_subway_1km"] + df["num_bus_1km"]
    return df

# 면적 구간화 함수 (소형/중형/대형/초대형)
def bin_area(x):
    if x < 60: return '소형'
    elif x < 85: return '중형'
    elif x < 135: return '대형'
    else: return '초대형'

# 층수 구간화 함수 (저층/중층/고층)
def bin_floor(x):
    if pd.isna(x): return '미상'
    if x <= 5: return '저층'
    elif x <= 15: return '중층'
    else: return '고층'

# 대장 아파트 거리 기준 플래그 생성 (100m 이내 여부)
def add_leading_apt_flags(df, distance_col="대장아파트거리"):
    df["대장_근접여부_100m"] = (df[distance_col] <= 100).astype(int)

# 데이터 로딩 함수 (경로별 데이터 불러오기)
def load_data(train_path, test_path, bus_path, subway_path, submission_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    bus = pd.read_csv(bus_path)
    subway = pd.read_csv(subway_path)
    submission = pd.read_csv(submission_path)
    return train, test, bus, subway, submission

# 전체 전처리 로직 (train + test 결합 후 처리)
def preprocess_enhanced(train, test, bus, subway, radius_km=1.0):
    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = np.nan
    combined = pd.concat([train, test], axis=0)

    # 계약일 관련 파생 컬럼 생성
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

    # 컬럼별 TAG 처리 (drop, impute, keep 등)
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

    # 파생 변수 생성
    if "건축년도" in combined.columns:
        combined["building_age"] = (CURRENT_YEAR - combined["건축년도"]).clip(lower=0)

    if "전용면적(㎡)" in combined.columns:
        combined["전용면적_bin"] = combined["전용면적(㎡)"].apply(bin_area).astype("category").cat.codes

    if "층" in combined.columns:
        combined["층수_bin"] = combined["층"].apply(bin_floor).astype("category").cat.codes
        if "전용면적(㎡)" in combined.columns:
            combined["floor_x_area"] = combined["층"] * combined["전용면적(㎡)"]
        combined["층수_면적_비율"] = combined["층"] / (combined["전용면적(㎡)"] + 1e-3)

    # 강남 중심 거리 계산
    if {"좌표X", "좌표Y"}.issubset(combined.columns):
        combined["distance_from_gangnam_center"] = np.sqrt(
            (combined["좌표X"] - GANGNAM_CENTER_X) ** 2 + (combined["좌표Y"] - GANGNAM_CENTER_Y) ** 2
        )

    # 교통 밀도 파생 (1km 기준)
    combined = add_transport_features(combined, bus, subway, radius_km)

    # 복합 키 및 대장 관련 변수
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

    # 학습/테스트 분리
    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])

    # 타겟 인코딩 적용
    for col in ["전용면적_bin", "시군구_아파트명"]:
        if col in train_processed.columns:
            train_processed, test_processed = apply_target_encoding(train_processed, test_processed, col)
            train_processed.drop(columns=col, inplace=True)
            test_processed.drop(columns=col, inplace=True)

    return train_processed, test_processed, bus, subway
