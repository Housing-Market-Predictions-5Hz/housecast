#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
교통 특성 처리 및 좌표 결측치 보간 테스트 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data
from preprocessor import preprocess_data, add_transport_features, build_ball_tree, coord_cols, boost_coordinates

def test_coordinate_imputation():
    """좌표 결측치 보간 테스트"""
    print("\n=== 좌표 결측치 보간 테스트 ===")
    
    # 데이터 로드
    train_data, test_data, bus_data, subway_data = load_data()
    
    # 샘플링 (빠른 테스트를 위해)
    train_sample = train_data.sample(n=1000, random_state=42)
    
    # 일부 좌표 값을 인위적으로 NaN으로 만들기
    train_sample.loc[train_sample.sample(n=200, random_state=42).index, '좌표X'] = np.nan
    train_sample.loc[train_sample.sample(n=200, random_state=42).index, '좌표Y'] = np.nan
    
    # 결측치 수 확인
    print(f"X 좌표 결측치: {train_sample['좌표X'].isna().sum()}/{len(train_sample)}")
    print(f"Y 좌표 결측치: {train_sample['좌표Y'].isna().sum()}/{len(train_sample)}")
    
    # boost_coordinates 함수를 사용하여 결측치 보간
    train_sample = boost_coordinates(train_sample)
    
    # 결과 확인
    print(f"보간 후 X 좌표 결측치: {train_sample['좌표X'].isna().sum()}/{len(train_sample)}")
    print(f"보간 후 Y 좌표 결측치: {train_sample['좌표Y'].isna().sum()}/{len(train_sample)}")
    
    # 좌표 분포 확인
    print(f"좌표X 분포: 최소={train_sample['좌표X'].min():.2f}, 최대={train_sample['좌표X'].max():.2f}, 평균={train_sample['좌표X'].mean():.2f}")
    print(f"좌표Y 분포: 최소={train_sample['좌표Y'].min():.2f}, 최대={train_sample['좌표Y'].max():.2f}, 평균={train_sample['좌표Y'].mean():.2f}")
    
    return train_sample

def test_transport_features():
    """교통 특성 추가 테스트"""
    print("\n=== 교통 특성 추가 테스트 ===")
    
    # 데이터 로드
    train_data, test_data, bus_data, subway_data = load_data()
    
    # 샘플링 (빠른 테스트를 위해)
    train_sample = train_data.sample(n=500, random_state=42)
    
    # 결측치 수 확인
    print(f"X 좌표 결측치: {train_sample['좌표X'].isna().sum()}/{len(train_sample)}")
    print(f"Y 좌표 결측치: {train_sample['좌표Y'].isna().sum()}/{len(train_sample)}")
    
    # boost_coordinates 함수를 사용하여 모든 결측치 보간
    if train_sample['좌표X'].isna().sum() > 0 or train_sample['좌표Y'].isna().sum() > 0:
        print("좌표 결측치가 있어 보간을 진행합니다...")
        train_sample = boost_coordinates(train_sample)
    
    # BallTree 구축
    try:
        bus_lon, bus_lat = coord_cols(bus_data)
        subway_lon, subway_lat = coord_cols(subway_data)
        
        tree_bus = build_ball_tree(bus_data)
        tree_subway = build_ball_tree(subway_data)
        
        # 교통 특성 추가
        train_sample = add_transport_features(train_sample, tree_subway, tree_bus)
        
        # 결과 확인 - 5개 교통 특성
        transport_cols = ['dist_subway_min', 'cnt_subway_500', 'cnt_subway_1000', 'dist_bus_min', 'cnt_bus_300']
        print("\n교통 특성 통계:")
        
        for col in transport_cols:
            print(f"{col}:")
            print(f"  - 결측치: {train_sample[col].isna().sum()}")
            print(f"  - 평균: {train_sample[col].mean():.2f}")
            print(f"  - 최소: {train_sample[col].min():.2f}")
            print(f"  - 최대: {train_sample[col].max():.2f}")
        
        # 컬럼이 모두 있는지 확인
        missing_cols = [col for col in transport_cols if col not in train_sample.columns]
        if missing_cols:
            print(f"경고: 누락된 교통 특성이 있습니다: {missing_cols}")
        else:
            print("✓ 모든 교통 특성이 성공적으로 추가되었습니다.")
            
        # 교통 특성 간 상관관계 분석
        corr = train_sample[transport_cols].corr()
        print("\n교통 특성 간 상관관계:")
        print(corr.round(2))
        
    except Exception as e:
        print(f"교통 특성 추가 중 오류 발생: {e}")
    
    return train_sample

def run_full_pipeline_test():
    """전체 파이프라인 테스트"""
    print("\n=== 전체 파이프라인 테스트 ===")
    
    # 데이터 로드
    train_data, test_data, bus_data, subway_data = load_data()
    
    # 샘플링 (빠른 테스트를 위해)
    train_sample = train_data.sample(n=1000, random_state=42)
    test_sample = test_data.sample(n=500, random_state=42)
    
    try:
        # 전처리 수행
        train_processed, test_processed, _, _ = preprocess_data(train_sample, test_sample, bus_data, subway_data)
        
        # 특성 목록 확인
        train_cols = train_processed.columns.tolist()
        test_cols = test_processed.columns.tolist()
        
        print(f"훈련 데이터 형태: {train_processed.shape}")
        print(f"테스트 데이터 형태: {test_processed.shape}")
        
        # 교통 관련 특성 확인
        transport_cols = [col for col in train_cols if any(substr in col for substr in 
                        ['subway', 'dist_', 'cnt_', 'bus'])]
        
        print("\n교통 관련 특성:")
        for col in transport_cols:
            print(f"  - {col}")
        
        # 필수 교통 특성이 모두 있는지 확인
        required_cols = ['dist_subway_min', 'cnt_subway_500', 'cnt_subway_1000', 'dist_bus_min', 'cnt_bus_300']
        missing_cols = [col for col in required_cols if col not in train_cols]
        
        if missing_cols:
            print(f"경고: 누락된 필수 교통 특성이 있습니다: {missing_cols}")
        else:
            print("✓ 모든 교통 특성이 성공적으로 추가되었습니다.")
            
        # 모든 특성의 결측치 확인
        na_counts = train_processed.isna().sum()
        cols_with_na = na_counts[na_counts > 0]
        if not cols_with_na.empty:
            print("\n경고: 결측치가 있는 특성이 있습니다:")
            for col, count in cols_with_na.items():
                print(f"  - {col}: {count}개 결측치")
        else:
            print("✓ 모든 특성에 결측치가 없습니다.")
            
    except Exception as e:
        print(f"전체 파이프라인 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    print("교통 특성 처리 및 좌표 결측치 보간 테스트 시작")
    
    # 좌표 결측치 보간 테스트
    train_with_coords = test_coordinate_imputation()
    
    # 교통 특성 추가 테스트
    train_with_transport = test_transport_features()
    
    # 전체 파이프라인 테스트
    run_full_pipeline_test()
    
    print("\n테스트 완료!") 