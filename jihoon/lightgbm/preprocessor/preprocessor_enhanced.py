import pandas as pd
import numpy as np
from preprocessor.column_tags import TAGS

CURRENT_YEAR = 2023
GANGNAM_CENTER_X = 203731
GANGNAM_CENTER_Y = 452331

# 외부 단지코드 불러오기
def load_complex_code(path="data/Complex Code_20221129.xlsx"):
    df = pd.read_excel(path)
    df["매칭키"] = df["시군구"].astype(str) + "_" + df["건물명"].astype(str)
    return df[["매칭키", "단지코드"]].dropna()

# 전용면적 구간화
def bin_area(x):
    if x < 60: return '소형'
    elif x < 85: return '중형'
    elif x < 135: return '대형'
    else: return '초대형'

# 층수 구간화
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
    sample_submission = pd.read_csv(submission_path)
    return train, test, bus, subway, sample_submission

def preprocess_data(train, test, bus, subway):
    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = np.nan
    combined = pd.concat([train, test], axis=0)

    # ✅ 외부 단지코드 병합
    complex_map = load_complex_code()
    combined["매칭키"] = combined["시군구"].astype(str) + "_" + combined["아파트명"].astype(str)
    combined = pd.merge(combined, complex_map, how="left", on="매칭키")

    # ✅ 단지코드 존재 확인
    if "단지코드" not in combined.columns:
        raise KeyError("❌ '단지코드' 컬럼이 누락되어 전처리를 진행할 수 없습니다. 외부 매핑 확인 필요")

    for col, tag in TAGS.items():
        if col not in combined.columns:
            continue
        if tag == "drop":
            combined.drop(columns=col, inplace=True)
        elif tag == "flag":
            combined[f"is_na_{col}"] = combined[col].isnull().astype(int)
            combined.drop(columns=col, inplace=True)
        elif tag == "impute":
            if combined[col].dtype == "object":
                combined[col].fillna("미상", inplace=True)
                combined[col] = combined[col].astype("category").cat.codes
            else:
                combined[col].fillna(combined[col].median(), inplace=True)
        elif tag == "coord":
            combined[col].fillna(0, inplace=True)
        elif tag == "categorical":
            combined[col] = combined[col].fillna("미상")
            combined[col] = combined[col].astype("category").cat.codes
        elif tag == "keep":
            combined[col].fillna("미상" if combined[col].dtype == "object" else combined[col].median(), inplace=True)

    if "건축년도" in combined.columns:
        combined["building_age"] = CURRENT_YEAR - combined["건축년도"]
        combined["building_age"] = combined["building_age"].clip(lower=0)

    if "전용면적(㎡)" in combined.columns:
        combined["전용면적_bin"] = combined["전용면적(㎡)"].apply(bin_area)
        combined["전용면적_bin"] = combined["전용면적_bin"].astype("category").cat.codes

    if "층" in combined.columns:
        combined["층수_bin"] = combined["층"].apply(bin_floor)
        combined["층수_bin"] = combined["층수_bin"].astype("category").cat.codes
        if "전용면적(㎡)" in combined.columns:
            combined["floor_x_area"] = combined["층"] * combined["전용면적(㎡)"]

    if "좌표X" in combined.columns and "좌표Y" in combined.columns:
        combined["distance_from_gangnam_center"] = np.sqrt(
            (combined["좌표X"] - GANGNAM_CENTER_X) ** 2 + (combined["좌표Y"] - GANGNAM_CENTER_Y) ** 2
        )

    if "시군구" in combined.columns and "단지코드" in combined.columns:
        combined["시군구_단지"] = combined["시군구"].astype(str) + "_" + combined["단지코드"].astype(str)
        combined["시군구_단지"] = combined["시군구_단지"].astype("category").cat.codes

    if "target" in combined.columns:
        mean_target_by_apt = combined[combined['is_train'] == 1].groupby("단지코드")["target"].mean()
        combined["단지별_평균가"] = combined["단지코드"].map(mean_target_by_apt)
        combined["단지별_평균가"].fillna(combined["단지별_평균가"].median(), inplace=True)

    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])

    bus.fillna(0, inplace=True)
    subway.fillna(0, inplace=True)

    return train_processed, test_processed, bus, subway
