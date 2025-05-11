import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from preprocessor.column_tags import TAGS

CURRENT_YEAR = 2023
GANGNAM_CENTER_X = 203731
GANGNAM_CENTER_Y = 452331

LEAD_HOUSE_COORDS = {
    "강서구": (37.5652, 126.8235), "관악구": (37.4780, 126.9418), "강남구": (37.5306, 127.0263),
    "강동구": (37.5572, 127.1636), "광진구": (37.5431, 127.0998), "구로구": (37.5105, 126.8869),
    "금천구": (37.4598, 126.8974), "노원구": (37.6395, 127.0723), "도봉구": (37.6578, 127.0435),
    "동대문구": (37.5776, 127.0538), "동작구": (37.5099, 126.9618), "마포구": (37.5434, 126.9360),
    "서대문구": (37.5581, 126.9559), "서초구": (37.5063, 126.9985), "성동구": (37.5387, 127.0450),
    "성북구": (37.6116, 127.0270), "송파구": (37.5128, 127.0834), "양천구": (37.5268, 126.8662),
    "영등포구": (37.5207, 126.9367), "용산구": (37.5212, 126.9735), "은평구": (37.6018, 126.9363),
    "종로구": (37.5686, 126.9669), "중구": (37.5545, 126.9635), "중랑구": (37.5817, 127.0818),
    "강북구": (37.6119, 127.0282)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

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

    for col, tag in TAGS.items():
        if col not in combined.columns:
            continue
        if tag == "drop":
            combined.drop(columns=col, inplace=True)
        elif tag == "flag":
            combined[f"is_na_{col}"] = combined[col].isnull().astype(int)
            combined.drop(columns=col, inplace=True)
        elif tag == "impute":
            if combined[col].dtype == 'object':
                combined[col].fillna("미상", inplace=True)
            else:
                combined[col].fillna(combined[col].median(), inplace=True)
        elif tag == "coord":
            combined[col].fillna(0, inplace=True)
        elif tag == "categorical":
            combined[col] = combined[col].fillna("미상")
            combined[col] = combined[col].astype("category").cat.codes

    if "좌표X" in combined.columns and "좌표Y" in combined.columns:
        combined["distance_from_gangnam_center"] = np.sqrt(
            (combined["좌표X"] - GANGNAM_CENTER_X) ** 2 + (combined["좌표Y"] - GANGNAM_CENTER_Y) ** 2
        )

    if "건축년도" in combined.columns:
        combined["building_age"] = CURRENT_YEAR - combined["건축년도"]
        combined["building_age"] = combined["building_age"].clip(lower=0)

    if "전용면적(㎡)" in combined.columns and "target" in combined.columns:
        combined["area_per_unit_price"] = combined["target"] / (combined["전용면적(㎡)"] + 1e-6)

    if "층" in combined.columns and "전용면적(㎡)" in combined.columns:
        combined["floor_x_area"] = combined["층"] * combined["전용면적(㎡)"]

    if "시군구" in combined.columns:
        lead_df = pd.DataFrame([
            {"시군구": k, "대장Y": v[0], "대장X": v[1]} for k, v in LEAD_HOUSE_COORDS.items()
        ])
        combined = pd.merge(combined, lead_df, how="left", on="시군구")
        combined["distance_from_leading_apt"] = combined.apply(
            lambda row: haversine(row["좌표Y"], row["좌표X"], row["대장Y"], row["대장X"]), axis=1
        )
        combined.drop(columns=["대장Y", "대장X"], inplace=True)

    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])
    bus.fillna(0, inplace=True)
    subway.fillna(0, inplace=True)
    return train_processed, test_processed, bus, subway