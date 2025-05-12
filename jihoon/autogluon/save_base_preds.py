from autogluon.tabular import TabularPredictor
import pandas as pd
import os
from preprocessor.preprocessor_enhanced import load_data, preprocess_data

# 경로 설정
predictor_path = "output/autogluon"
output_dir = "base_preds"
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
predictor = TabularPredictor.load(predictor_path)
model_names = predictor.model_names()
print("📋 사용 가능한 모델 목록:")
print(model_names)

# 데이터 로드 및 전처리
train_path = "data/train.csv"
test_path = "data/test.csv"
bus_path = "data/bus_feature.csv"
subway_path = "data/subway_feature.csv"
submission_path = "data/sample_submission.csv"

train_df, test_df, bus_df, subway_df, _ = load_data(
    train_path, test_path, bus_path, subway_path, submission_path
)
_, test_processed, _, _ = preprocess_data(train_df, test_df, bus_df, subway_df)

# AutoGluon 사용 feature만 추출
X_test = test_processed[predictor.features()]

# 예측 저장 함수
def save_prediction(model_name, alias):
    if model_name not in model_names:
        print(f"⚠️ {model_name} 없음 → 건너뜀")
        return
    preds = predictor.predict(X_test, model=model_name, transform_features=False)
    preds.to_frame(name="target").to_csv(f"{output_dir}/{alias}_preds.csv", index_label="id")
    print(f"✅ {model_name} → {alias}_preds.csv 저장 완료")

# 예측 수행
save_prediction("LightGBM_BAG_L2", "lgbm")
save_prediction("CatBoost_BAG_L2", "catboost")
save_prediction("XGBoost_BAG_L2", "xgboost")