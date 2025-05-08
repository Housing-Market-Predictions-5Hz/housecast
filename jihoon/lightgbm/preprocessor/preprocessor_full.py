import pandas as pd
import numpy as np
from preprocessor.column_tags import TAGS

def load_data(train_path, test_path, bus_path, subway_path, submission_path):
    # 결측 컬럼 dtype 경고 방지
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    bus = pd.read_csv(bus_path)
    subway = pd.read_csv(subway_path)
    sample_submission = pd.read_csv(submission_path)
    return train, test, bus, subway, sample_submission

def preprocess_data(train, test, bus, subway):
    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = np.nan  # 타겟 컬럼 통합 후 제거

    combined = pd.concat([train, test], axis=0)

    for col, tag in TAGS.items():
        if col not in combined.columns:
            continue

        if tag == "drop":
            combined.drop(columns=col, inplace=True)

        elif tag == "flag":
            combined[f"is_na_{col}"] = combined[col].isnull().astype(int)
            combined.drop(columns=col, inplace=True)  # 원본 제거

        elif tag == "impute":
            if combined[col].dtype == 'object':
                combined[col].fillna("미상", inplace=True)
                combined[col] = combined[col].astype("category").cat.codes  # LightGBM 호환
            else:
                combined[col].fillna(combined[col].median(), inplace=True)

        elif tag == "coord":
            combined[col].fillna(0, inplace=True)

        elif tag == "categorical":
            combined[col] = combined[col].fillna("미상")
            combined[col] = combined[col].astype("category").cat.codes

        # 'keep'은 유지

    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])

    bus.fillna(0, inplace=True)
    subway.fillna(0, inplace=True)

    return train_processed, test_processed, bus, subway