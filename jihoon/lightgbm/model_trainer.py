from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import joblib

def train_model(X_train, y_train):
    print("LightGBM 모델 훈련 시작...")

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("모델 훈련 완료!")

    # 중요도 저장
    feature_importance = model.feature_importances_
    return model, feature_importance

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def save_model(model, output_dir="output", filename="model_lgbm.pkl"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    print(f"모델 저장 완료: {path}")
