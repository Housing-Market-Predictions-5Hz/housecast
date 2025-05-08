import pandas as pd

# ✔️ 현재 디렉토리에 맞게 수정
df = pd.read_csv("train.csv")

# 결측률 분석 함수
def missing_report(df):
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    dtype = df.dtypes
    report = pd.DataFrame({
        "Missing": total,
        "Percent": percent,
        "Dtype": dtype
    })
    return report[report["Missing"] > 0].sort_values(by="Percent", ascending=False)

# 실행
report = missing_report(df)
print(report)
