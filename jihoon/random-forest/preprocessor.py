#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - Random Forest 모델
전처리 모듈

이 모듈은 데이터 전처리, 특성 엔지니어링 및 변환을 담당합니다.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(train_data, test_data, bus_data, subway_data):
    """
    데이터 전처리 및 특성 엔지니어링을 수행하는 함수
    
    Args:
        train_data (pd.DataFrame): 훈련 데이터셋
        test_data (pd.DataFrame): 테스트 데이터셋
        bus_data (pd.DataFrame): 버스 관련 데이터
        subway_data (pd.DataFrame): 지하철 관련 데이터
        
    Returns:
        tuple: (X_train, y_train, X_test, processed_data)
    """
    print("데이터 전처리 및 특성 엔지니어링 시작...")
    
    # 데이터 복사본 생성
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # 타겟 변수 분리
    y_train = train_df['target'].values
    
    # 트레인과 테스트 데이터 통합 (전처리를 위해)
    test_df['is_test'] = 1
    train_df['is_test'] = 0
    combined_df = pd.concat([train_df, test_df])
    
    # 컬럼명 간소화
    combined_df = combined_df.rename(columns={'전용면적(㎡)': '전용면적'})
    
    # 1. 결측치 처리
    print("1. 결측치 처리 중...")
    
    # 불필요한 결측치가 많은 컬럼 확인 및 제거
    missing_ratio = combined_df.isna().sum() / combined_df.shape[0]
    high_missing_cols = missing_ratio[missing_ratio > 0.9].index.tolist()
    print(f"   - 결측치가 90% 이상인 컬럼 {len(high_missing_cols)}개 제거")
    combined_df = combined_df.drop(columns=high_missing_cols)
    
    # 범주형 변수와 수치형 변수 구분
    numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    
    # 특수 결측치 처리 (공백이나 '-' 등)
    combined_df['등기신청일자'] = combined_df['등기신청일자'].replace(' ', np.nan)
    combined_df['거래유형'] = combined_df['거래유형'].replace('-', np.nan)
    combined_df['중개사소재지'] = combined_df['중개사소재지'].replace('-', np.nan)
    
    # 범주형 변수 결측치 처리
    for col in categorical_cols:
        combined_df[col] = combined_df[col].fillna('NULL')
    
    # 수치형 변수 결측치 처리 (선형 보간)
    for col in numeric_cols:
        if combined_df[col].isna().sum() > 0:
            combined_df[col] = combined_df[col].interpolate(method='linear')
    
    # 2. 이상치 처리
    print("2. 이상치 처리 중...")
    
    # IQR 방식으로 이상치 감지 및 처리
    for col in numeric_cols:
        if col not in ['target', 'is_test', 'id']:
            Q1 = combined_df[col].quantile(0.25)
            Q3 = combined_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치를 경계값으로 대체
            combined_df[col] = np.where(combined_df[col] < lower_bound, lower_bound, combined_df[col])
            combined_df[col] = np.where(combined_df[col] > upper_bound, upper_bound, combined_df[col])
    
    # 3. 특성 엔지니어링
    print("3. 특성 엔지니어링 중...")
    
    # 날짜 관련 특성 추출 (계약일)
    if '계약일' in combined_df.columns:
        # 결측값이나 빈 문자열 처리
        combined_df['계약일'] = combined_df['계약일'].fillna('').astype(str)
        
        # 계약년도 추출 (빈 문자열 처리)
        combined_df['계약년도'] = combined_df['계약일'].apply(
            lambda x: float(x[:4]) if x and len(x) >= 4 and x[:4].isdigit() else np.nan
        )
        
        # 계약월 추출 (빈 문자열 처리)
        combined_df['계약월'] = combined_df['계약일'].apply(
            lambda x: float(x[4:6]) if x and len(x) >= 6 and x[4:6].isdigit() else np.nan
        )
        
        # 계약계절 계산 (결측값 처리)
        combined_df['계약계절'] = combined_df['계약월'].apply(
            lambda x: (int(x)%12+3)//3 if not pd.isna(x) else np.nan
        )
    
    # 층수 관련 특성 (저층/중층/고층)
    if '층' in combined_df.columns:
        combined_df['층_구분'] = pd.cut(combined_df['층'], 
                                     bins=[-float('inf'), 3, 10, float('inf')], 
                                     labels=['저층', '중층', '고층'])
    
    # 면적에 따른 평수 계산
    if '전용면적' in combined_df.columns:
        combined_df['평수'] = combined_df['전용면적'] * 0.3025
    
    # 건물 연식 계산
    if '건축년도' in combined_df.columns and '계약년도' in combined_df.columns:
        combined_df['건물연식'] = combined_df['계약년도'] - combined_df['건축년도']
    
    # 버스 정류장 개수와 최소 거리 추가
    if 'id' in combined_df.columns and 'id' in bus_data.columns:
        bus_counts = bus_data.groupby('id').size().reset_index(name='버스정류장수')
        bus_min_dist = bus_data.groupby('id')['거리'].min().reset_index(name='버스최소거리')
        
        combined_df = pd.merge(combined_df, bus_counts, on='id', how='left')
        combined_df = pd.merge(combined_df, bus_min_dist, on='id', how='left')
        combined_df['버스정류장수'] = combined_df['버스정류장수'].fillna(0)
        combined_df['버스최소거리'] = combined_df['버스최소거리'].fillna(combined_df['버스최소거리'].max())
    
    # 지하철역 개수와 최소 거리 추가
    if 'id' in combined_df.columns and 'id' in subway_data.columns:
        subway_counts = subway_data.groupby('id').size().reset_index(name='지하철역수')
        subway_min_dist = subway_data.groupby('id')['거리'].min().reset_index(name='지하철최소거리')
        
        combined_df = pd.merge(combined_df, subway_counts, on='id', how='left')
        combined_df = pd.merge(combined_df, subway_min_dist, on='id', how='left')
        combined_df['지하철역수'] = combined_df['지하철역수'].fillna(0)
        combined_df['지하철최소거리'] = combined_df['지하철최소거리'].fillna(combined_df['지하철최소거리'].max())
    
    # 4. 인코딩 및 스케일링
    print("4. 인코딩 및 스케일링 중...")
    
    # 라벨 인코딩 (범주형 -> 수치형)
    label_encoders = {}
    for col in categorical_cols:
        if col not in ['id', 'target']:
            # 모든 값을 문자열로 변환하여 타입 일관성 보장
            combined_df[col] = combined_df[col].astype(str)
            le = LabelEncoder()
            combined_df[col] = le.fit_transform(combined_df[col])
            label_encoders[col] = le
    
    # 필요없는 컬럼 제거
    drop_cols = ['id', 'target', '계약일', '등기신청일자']
    drop_cols = [col for col in drop_cols if col in combined_df.columns]
    combined_df = combined_df.drop(columns=drop_cols)
    
    # 스케일링 (수치형 변수 정규화)
    scaler = StandardScaler()
    
    # 재정의된 numeric_cols (특성 엔지니어링 후 추가된 컬럼 포함)
    numeric_cols = [col for col in combined_df.columns if col not in ['is_test', '층_구분'] 
                    and combined_df[col].dtype != 'object' 
                    and pd.api.types.is_numeric_dtype(combined_df[col])]
    
    # 원-핫 인코딩이 필요한 범주형 컬럼 처리
    if '층_구분' in combined_df.columns:
        층_구분_dummies = pd.get_dummies(combined_df['층_구분'], prefix='층_구분')
        combined_df = pd.concat([combined_df, 층_구분_dummies], axis=1)
        combined_df = combined_df.drop('층_구분', axis=1)
    
    # 스케일링 적용
    combined_df[numeric_cols] = scaler.fit_transform(combined_df[numeric_cols])
    
    # 데이터 분리
    train_processed = combined_df[combined_df['is_test'] == 0].drop('is_test', axis=1)
    test_processed = combined_df[combined_df['is_test'] == 1].drop('is_test', axis=1)

    # NaN 제거
    train_processed = train_processed.fillna(0)
    test_processed = test_processed.fillna(0)

    X_train = train_processed.values
    X_test = test_processed.values
    
    # 최종 정보
    print(f"전처리 완료: X_train 형태 {X_train.shape}, X_test 형태 {X_test.shape}")
    
    # 디버깅 및 분석을 위한 처리된 데이터
    processed_data = {
        'X_train': pd.DataFrame(X_train, columns=train_processed.columns),
        'X_test': pd.DataFrame(X_test, columns=test_processed.columns),
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    
    return X_train, y_train, X_test, processed_data

if __name__ == "__main__":
    # 테스트 실행을 위한 데이터 로드
    from data_loader import load_data
    train_data, test_data, bus_data, subway_data = load_data()
    
    # 전처리 테스트
    X_train, y_train, X_test, processed_data = preprocess_data(train_data, test_data, bus_data, subway_data)
    
    # 결과 확인
    print("\n전처리 결과 확인:")
    print(f"X_train 형태: {X_train.shape}")
    print(f"y_train 형태: {y_train.shape}")
    print(f"X_test 형태: {X_test.shape}")
    print("\n특성 목록:")
    print(processed_data['X_train'].columns.tolist()) 
