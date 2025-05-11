import pandas as pd
import numpy as np

# 데이터 불러오기
train = pd.read_csv("../data/train.csv", low_memory=False)

# 숫자형 컬럼만 추출
numeric_cols = train.select_dtypes(include=[np.number])

# 상관계수 행렬 계산
corr_matrix = numeric_cols.corr()

# target과의 상관계수 추출 및 정렬
correlation_with_target = corr_matrix["target"].sort_values(ascending=False)
print("\n🎯 Target과의 상관계수 Top 20:\n")
print(correlation_with_target.head(20))

# 저분산 컬럼 찾기 (분산이 너무 작은 컬럼은 정보량이 적음)
low_variance_cols = numeric_cols.var()[numeric_cols.var() < 1e-2].index.tolist()
print("\n⚠️ 저분산 컬럼 목록 (정보량이 적어 제거 고려):\n")
print(low_variance_cols)