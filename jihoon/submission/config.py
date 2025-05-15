# config.py
# 전체 프로젝트에서 공통으로 사용할 설정값을 정의한 모듈

import os

# =========================
# [1] 경로 설정
# =========================

# 원본 데이터가 저장된 디렉토리
DATA_DIR = "data"

# 결과 및 제출 파일 저장 디렉토리
OUTPUT_DIR = "output/final_submission_top25"

# 최종 학습된 모델 파일 경로
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_lgbm_optuna_top25.pkl")

# 컬럼명 정제 정보 저장 파일 경로 (clean_column_names에서 생성됨)
COLUMN_MAP_PATH = os.path.join(OUTPUT_DIR, "column_name_map.csv")

# 샘플 제출 파일 경로
SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")


# =========================
# [2] 전처리 및 학습 설정
# =========================

# 교통 밀도 계산 시 사용할 반경 (단위: km) — 1.0km로 고정
RADIUS_KM = 1.0

# 예측에 사용할 feature 개수 (Top-N 중요도 기준)
TOP_N = 25