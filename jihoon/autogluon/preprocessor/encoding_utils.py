# encoding_utils.py

import pandas as pd
import numpy as np

def frequency_encode(df: pd.DataFrame, column: str) -> pd.Series:
    freq = df[column].value_counts()
    return df[column].map(freq)

def target_encode(df: pd.DataFrame, column: str, target: pd.Series) -> pd.Series:
    # 각 범주의 평균 target 값 계산
    target_mean = df.groupby(column)[target.name].mean()
    return df[column].map(target_mean)