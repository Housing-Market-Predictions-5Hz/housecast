import pandas as pd
import numpy as np
import os
from pathlib import Path
from preprocessor.preprocessor_enhanced import boost_coordinates, load_apt_coordinates

def load_data(train_path, test_path):
    """
    데이터셋 로드 함수
    """
    print(f"Train 데이터 로드 중: {train_path}")
    train = pd.read_csv(train_path, low_memory=False)
    
    print(f"Test 데이터 로드 중: {test_path}")
    test = pd.read_csv(test_path, low_memory=False)
    
    print(f"Train 크기: {train.shape}, Test 크기: {test.shape}")
    return train, test

def impute_coordinates_and_save(train_df, test_df, output_dir="output"):
    """
    좌표 결측치 보간 후 파일 저장
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 좌표 결측치 상태 확인
    print("\n=== 보간 전 좌표 결측치 상태 ===")
    for name, df in [("Train", train_df), ("Test", test_df)]:
        for col in ["좌표X", "좌표Y"]:
            if col in df.columns:
                missing = df[col].isna().sum()
                missing_pct = missing / len(df) * 100
                print(f"{name} {col} 결측치: {missing}/{len(df)} ({missing_pct:.2f}%)")
    
    # 데이터셋 합치기 (보간을 위해)
    print("\n좌표 보간을 위해 train/test 데이터셋 결합...")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    
    # test에는 target 컬럼이 없을 수 있으므로 추가
    if "target" not in test_df.columns:
        test_df["target"] = np.nan
    
    combined = pd.concat([train_df, test_df], axis=0)
    print(f"결합된 데이터셋 크기: {combined.shape}")
    
    # 좌표 결측치 보간
    print("\n좌표 결측치 보간 시작...")
    combined = boost_coordinates(combined)
    
    # 보간 후 결측치 확인
    print("\n=== 보간 후 좌표 결측치 상태 ===")
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            missing = combined[col].isna().sum()
            missing_pct = missing / len(combined) * 100
            print(f"전체 {col} 결측치: {missing}/{len(combined)} ({missing_pct:.2f}%)")
    
    # 다시 분리
    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])
    
    # 좌표 통계
    print("\n=== 보간된 좌표 통계 ===")
    for name, df in [("Train", train_processed), ("Test", test_processed)]:
        for col in ["좌표X", "좌표Y"]:
            if col in df.columns:
                print(f"{name} {col} - 최소: {df[col].min():.6f}, 최대: {df[col].max():.6f}, 평균: {df[col].mean():.6f}")
    
    # 파일 저장
    train_output_path = os.path.join(output_dir, "train_with_coordinates.csv")
    test_output_path = os.path.join(output_dir, "test_with_coordinates.csv")
    
    print(f"\n보간된 데이터 저장 중...")
    train_processed.to_csv(train_output_path, index=False)
    test_processed.to_csv(test_output_path, index=False)
    
    print(f"저장 완료:")
    print(f"  - Train: {train_output_path}")
    print(f"  - Test: {test_output_path}")
    
    return train_processed, test_processed

if __name__ == "__main__":
    # 경로 설정
    data_dir = Path("../../raw")  
    output_dir = "output"
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    # 데이터 로드
    train_df, test_df = load_data(train_path, test_path)
    
    # 좌표 보간 및 저장
    train_processed, test_processed = impute_coordinates_and_save(train_df, test_df, output_dir)
    
    print("\n모든 처리가 완료되었습니다!") 