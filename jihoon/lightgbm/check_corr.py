import pandas as pd
from preprocessor.preprocessor_enhanced import load_data, preprocess_enhanced

# 파일 경로
train_path = "data/train.csv"
test_path = "data/test.csv"
bus_path = "data/bus_feature.csv"
subway_path = "data/subway_feature.csv"
submission_path = "data/sample_submission.csv"

# 전처리 수행
train_df, test_df, bus_df, subway_df, submission = load_data(
    train_path, test_path, bus_path, subway_path, submission_path
)
train_processed, _, _, _ = preprocess_enhanced(train_df, test_df, bus_df, subway_df)

# 학습 데이터만 추출
train_processed = train_processed[train_processed["target"].notnull()]

# 수치형 컬럼 기준 상관계수
corr_matrix = train_processed.corr(numeric_only=True)
target_corr = corr_matrix["target"].drop("target").sort_values(ascending=False)

# 결과 출력
print("📊 파생 Feature 포함 상관계수 (상위 20개):")
print(target_corr.head(20))

# (선택) 결과 저장
target_corr.head(20).to_csv("output/top20_corr_features.csv")