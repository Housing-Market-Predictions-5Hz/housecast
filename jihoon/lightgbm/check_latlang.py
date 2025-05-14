import pandas as pd

train = pd.read_csv("data/train.csv", low_memory=False)
test = pd.read_csv("data/test.csv", low_memory=False)

print("📊 Train lat/lng 결측:")
print(train[["lat", "lng"]].isna().sum())

print("\n📊 Test lat/lng 결측:")
print(test[["lat", "lng"]].isna().sum())