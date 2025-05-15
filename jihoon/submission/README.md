# House Price Prediction - Optuna + LightGBM (Top-N Feature Ensemble)

이 프로젝트는 서울시 아파트 실거래가 데이터를 기반으로, 주요 변수(feature)를 선택하여 **Optuna를 활용한 LightGBM 하이퍼파라미터 튜닝 및 예측 모델링**을 수행한 회귀 분석 프로젝트입니다.  
모델 성능과 해석력을 함께 고려하며, **상위 25개 중요 변수만을 활용한 경량화 모델링**을 통해 최적 예측 결과를 생성합니다.

---

## 📌 모델 개요

- **모델 종류**: LightGBM (GBDT 기반 회귀 모델)  
- **튜닝 기법**: Optuna (Bayesian Optimization)  
- **검증 방식**: 5-Fold KFold 평균 RMSE  
- **Ensemble**: Fold별 예측 결과 평균 앙상블  
- **예측 대상(Target)**: `target` 컬럼 (실거래가) → `log1p` 변환 후 학습, `expm1`로 복원  

---

## 🧼 주요 전처리 내용

- **결측값 처리**  
  `column_tags.py`에 정의된 정책(`drop`, `impute`, `flag`, `keep`, `coord`, `categorical`)에 따라 컬럼별 자동 처리

- **시간 파생 변수 생성**  
  - `계약_월`, `계약_연`, `계약_계절`, `계약_일자`  
  - `계약년월` + `계약일` → `datetime` 변환

- **거리 기반 파생 변수**  
  - `대장아파트거리`, `distance_from_gangnam_center`  
  - `대장_근접여부_100m`, `대장_거리_log` 등

- **교통 밀도 변수 (반경 1.0km 기준)**  
  - `num_subway_1km`: 반경 1.0km 이내 지하철역 수  
  - `num_bus_1km`: 반경 1.0km 이내 버스정류장 수  
  - `교통_총밀도_1km`: 지하철 + 버스 합계

- **면적·층수 파생**  
  - `전용면적_bin`: 소형/중형/대형 구간화 후 인코딩  
  - `층수_bin`, `floor_x_area`, `층수_면적_비율`

- **Target Encoding 적용**  
  - `전용면적_bin`, `시군구_아파트명` 컬럼에 평균 target 기반 인코딩(`_te` 접미사)

---

## 🧪 실험 조건

| 항목                | 값 |
|---------------------|------------------|
| Optuna Trial 수     | 50회             |
| Feature 수          | Top 25개         |
| 거리 기준           | 반경 1.0km        |
| 성능 검증 방식      | 5-Fold 평균 RMSE |
| 제출 방식           | `expm1` → `np.round()` → Int 저장 |

---

## 📈 성능 요약

| 모델 버전            | RMSE (평균)     |
|----------------------|-----------------|
| Optuna Best RMSE     | 예: 5291.7378    |
| Top 25 재학습 모델    | 예: 5283.2441    |

※ 자세한 결과는 `optuna_trials.csv` 및 KFold 평균 로그 참고

---

## 🗂️ 주요 구성 파일

| 파일명                  | 설명 |
|-------------------------|------|
| `main_top25.py`         | 전체 학습 및 제출까지 수행하는 메인 스크립트 |
| `config.py`             | 경로, radius, top_n 등 설정값 관리 |
| `preprocessor_enhanced.py` | 전처리 로직 및 파생변수 생성 포함 |
| `column_tags.py`        | 컬럼별 처리 정책 (`drop`, `impute`, `keep` 등) 정의 |
| `run_inference.py`      | 저장된 모델로 test셋 예측 및 제출 파일 생성 |
| `requirements.txt`      | 실행에 필요한 라이브러리 명시 |

---

## 📁 디렉토리 구조 예시

```
.
├── .gitignore
├── README.md
├── config.py
├── data
│   ├── train.csv
│   ├── test.csv
│   ├── bus_feature.csv
│   ├── subway_feature.csv
│   └── sample_submission.csv
├── main_top25.py
├── run_inference.py
├── preprocessor
│   ├── __init__.py
│   ├── column_tags.py
│   └── preprocessor_enhanced.py
├── output
│   └── optuna
│       ├── output_optuna_top25.csv
│       ├── feature_importance.csv
│       ├── model_lgbm_optuna_top25.pkl
│       ├── optuna_trials.csv
│       └── column_name_map.csv
└── requirements.txt
```

---

## 🛠️ 실행 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 전체 학습 및 제출 파일 생성
```bash
python main_top25.py
```

### 3. 저장된 모델로 테스트셋 예측만 수행
```bash
python run_inference.py
```

---

## ✅ 유의 사항

- 제출 파일은 반드시 `int` 타입으로 저장되어야 하며, 예측값은 `np.expm1()` 후 `np.round()` 처리 필요  
- 전체 데이터로부터 추출된 **상위 25개 변수만**을 사용해 재학습 및 예측  
- 교통 밀도 관련 변수(`num_subway_1km`, `num_bus_1km`, `교통_총밀도_1km`)는 반경 1.0km 기준으로 계산됨  
- 전처리 및 학습 시 컬럼명 정제(`clean_column_names`)와 동일한 방식으로 컬럼 매핑 정합성 유지 필요  

---

## ℹ️ 참고 사항

- 본 프로젝트는 AI Stages 기반 부동산 실거래가 예측 문제를 바탕으로 구성되었으며, 실제 데이터 기반 실험 결과를 토대로 작성되었습니다.
- 필요 시 `CLI 인자화`, `모델 앙상블`, `Stacking`, `AutoML` 등 확장 가능