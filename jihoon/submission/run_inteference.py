# run_inference.py
# 학습된 LightGBM 모델을 이용해 테스트 데이터 예측 및 제출 파일을 생성하는 스크립트

import pandas as pd
import numpy as np
import joblib
import os

from preprocessor.preprocessor_enhanced import load_data, preprocess_enhanced  # 전처리 함수
from train_optuna import clean_column_names  # 학습 시 사용한 컬럼 정제 함수 재사용

# ===== 경로 및 설정 =====
DATA_DIR = "data"  # 데이터 파일 위치
OUTPUT_DIR = "output/optuna"  # 결과 저장 디렉토리
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_lgbm_optuna_top25.pkl")  # 학습된 모델 경로
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.csv")  # 중요도 정보
COLUMN_MAP_PATH = os.path.join(OUTPUT_DIR, "column_name_map.csv")  # 컬럼명 매핑 정보
SUBMISSION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output_optuna_top25.csv")  # 최종 제출 파일 경로
TOP_N = 25  # 예측에 사용할 상위 중요도 feature 수

# ===== 데이터 로드 및 전처리 =====
train_df, test_df, bus_df, subway_df, submission = load_data(
    f"{DATA_DIR}/train.csv", f"{DATA_DIR}/test.csv",
    f"{DATA_DIR}/bus_feature.csv", f"{DATA_DIR}/subway_feature.csv",
    f"{DATA_DIR}/sample_submission.csv"
)

# 전처리 함수에서 radius_km=1.0 (1km 기준) 설정하여 교통 특성 계산
# train은 사용하지 않으므로 discard하고 test만 사용
_, test_processed, _, _ = preprocess_enhanced(train_df, test_df, bus_df, subway_df, radius_km=1.0)

# 학습 시와 동일하게 컬럼 정제(clean) 수행 (학습 시 저장된 컬럼명 기준)
test_processed = clean_column_names(test_processed)

# ===== Top-N 중요도 feature 기반 예측 =====
# 중요도 상위 TOP_N feature 불러오기
feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
selected_features = feature_importance["feature"].head(TOP_N).tolist()

# 테스트 데이터에서 해당 feature만 선택
X_test = test_processed[selected_features]

# 학습된 모델 로드
model = joblib.load(MODEL_PATH)

# 예측 수행 (log1p → expm1 변환으로 복원)
preds = np.expm1(model.predict(X_test))

# ===== 제출 파일 생성 =====
submission = submission.copy()
submission["target"] = np.round(preds).astype(int)  # 예측 결과 반올림 후 삽입
submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)  # 제출 파일 저장

print("최종 제출 파일 생성 완료:", SUBMISSION_OUTPUT_PATH)