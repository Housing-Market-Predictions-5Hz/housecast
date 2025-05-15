#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
좌표X와 좌표Y의 결측치를 확인하는 스크립트
"""

import pandas as pd
import numpy as np
from data_loader import load_data

# 데이터 로드
train_data, test_data, bus_data, subway_data = load_data()

# 훈련 데이터에서 좌표X와 좌표Y의 결측치 확인
print("\n훈련 데이터에서 좌표X와 좌표Y의 결측치 확인:")
print(f"훈련 데이터 총 행 수: {train_data.shape[0]}")

if '좌표X' in train_data.columns:
    x_missing = train_data['좌표X'].isna().sum()
    x_missing_pct = (x_missing / train_data.shape[0]) * 100
    print(f"좌표X 결측치 수: {x_missing} ({x_missing_pct:.2f}%)")
else:
    print("좌표X 컬럼이 존재하지 않습니다.")

if '좌표Y' in train_data.columns:
    y_missing = train_data['좌표Y'].isna().sum()
    y_missing_pct = (y_missing / train_data.shape[0]) * 100
    print(f"좌표Y 결측치 수: {y_missing} ({y_missing_pct:.2f}%)")
else:
    print("좌표Y 컬럼이 존재하지 않습니다.")

# 좌표X, 좌표Y 모두 결측인 데이터 확인
if '좌표X' in train_data.columns and '좌표Y' in train_data.columns:
    both_missing = train_data[train_data['좌표X'].isna() & train_data['좌표Y'].isna()].shape[0]
    both_missing_pct = (both_missing / train_data.shape[0]) * 100
    print(f"좌표X와 좌표Y 모두 결측인 데이터 수: {both_missing} ({both_missing_pct:.2f}%)")

# 좌표 데이터가 있는 샘플 확인 (있을 경우)
if '좌표X' in train_data.columns and '좌표Y' in train_data.columns:
    print("\n좌표가 있는 데이터 샘플 (5개):")
    print(train_data[['좌표X', '좌표Y']].dropna().head(5)) 