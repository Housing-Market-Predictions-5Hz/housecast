# 🏡 housecast
**데이터와 머신러닝을 활용한 주택 시장 트렌드 예측 프로젝트**

---

## 📖 프로젝트 소개
Housecast는 데이터 분석과 머신러닝 기법을 활용하여  
주택 시장의 트렌드와 주택 가격을 예측하는 프로젝트입니다.

---

## 📂 프로젝트 구조
```
housecast/
├── data/
│   ├── raw/          # 원본 데이터 저장
│   │   └── empty2.txt # Git 폴더 추적용 빈 파일
│   ├── processed/    # 전처리된 데이터 저장
│   │   └── empty.txt # Git 폴더 추적용 빈 파일
├── models/
│   └── empty3.txt     # 학습된 모델 저장 폴더
├── reports/
│   └── empty4.txt     # 분석 리포트 저장 폴더
├── notebooks/
│   └── eda_template.ipynb  # 탐색적 데이터 분석(EDA) 템플릿
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── model_trainer.py
│   └── visualization/
│       └── plot_functions.py
├── .github/
│   ├── workflows/
│   │   └── python-lint-test.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE.md
├── README.md
├── requirements.txt
└── TODO.md
```

---

## 🚀 시작하는 방법
1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/Housing-Market-Predictions-5Hz/housecast.git
   cd housecast
   ```

2. 필요한 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

3. 개발 워크플로우를 확인합니다:
   - PR을 작성하면 자동으로 Python 코드 스타일 검사(Lint)와 테스트(Pytest)가 실행됩니다.
   - 모든 PR은 **2명 이상 승인 + 테스트 통과**해야 `main`에 Merge할 수 있습니다.

4. 노트북을 실행하거나, `src/` 모듈을 활용하여 데이터 분석을 시작하세요!

---

## 🛠️ 주요 기능
- 데이터 로딩 및 전처리
- 특성(피처) 엔지니어링
- 머신러닝 모델 학습 및 평가
- 시각화 및 리포트 생성
- GitHub Actions 기반 코드 품질 관리 (Lint + Test)
- 안전한 GitHub 협업 환경 구축 (Branch Protection Rules 적용)

---

## 🎯 프로젝트 목표 (Project Goals)
- 데이터 기반 주택 가격 예측 모델 개발
- 다양한 머신러닝 알고리즘 성능 비교
- 설명 가능한 모델(Explainable AI, SHAP 등) 구축
- 결과를 기반으로 투자/정책 인사이트 도출
- 프로덕션 배포(모델 API 서비스)까지 이어지는 전체 파이프라인 구축

---

## 🛣️ 로드맵 (Roadmap)
- [x] 프로젝트 초기 셋업 및 폴더 구조 정리
- [x] 기본 데이터 로딩 및 EDA 템플릿 제작
- [x] GitHub 협업 규칙 및 워크플로우 구축
- [ ] 다양한 피처 엔지니어링 기법 적용
- [ ] 머신러닝 모델 비교 (예: XGBoost, LightGBM, RandomForest)
- [ ] 성능 최적화 및 하이퍼파라미터 튜닝
- [ ] SHAP 기반 피처 중요도 해석
- [ ] 모델 API 서버 배포 (FastAPI 또는 Flask)
- [ ] 배포 후 모니터링 및 리포트 자동화

---

## 📦 배포 방법 (Deployment Guide)
- 학습된 모델(`models/`)을 API 서버(FastAPI, Flask 등)로 배포할 수 있습니다.
- 또는 Jupyter Notebook을 통해 인사이트 분석 리포트를 생성할 수도 있습니다.

---

## 🛠️ 사용 예시 (Usage Example)
```python
from src.data.data_loader import load_data
from src.features.feature_engineering import create_features
from src.models.model_trainer import train_model

# 데이터 불러오기
df = load_data('data/raw/house_prices.csv')

# 특성 엔지니어링
df = create_features(df)

# 모델 훈련 및 저장
train_model(df, target_column='price', model_save_path='models/house_price_model.pkl')
```

---

## 📚 데이터셋 출처 (Data Sources)
- 기타 공개 부동산 데이터 소스 활용

---

## 🔥 추가 안내: GitHub 협업 규칙
- 모든 작업은 feature 브랜치를 생성한 후 진행합니다.
- PR 작성 시 자동으로 Lint 및 Test가 실행됩니다.
- PR은 반드시 **2명 이상 승인**을 받아야 Merge할 수 있습니다.
- CI 테스트를 통과하지 못하면 Merge할 수 없습니다.
- Branch Protection Rule이 적용되어 있어 main 브랜치를 안전하게 보호합니다.

---
