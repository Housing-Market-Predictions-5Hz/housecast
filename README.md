# 🏡 housecast

**데이터와 머신러닝을 활용한 주택 시장 트렌드 예측 프로젝트**

---

## 부동산 가격 예측 프로젝트

이 프로젝트는 다양한 머신러닝 모델을 사용하여 부동산 가격을 예측하는 시스템을 개발합니다. 각 모델은 개발자별 디렉토리 내에서 관리되며, 가상환경을 통해 독립적으로 실행됩니다.

## 프로젝트 구조

```
/
├── raw/                              # 원본 데이터 디렉토리
│   ├── train.csv                     # 훈련 데이터
│   ├── test.csv                      # 테스트 데이터
│   ├── bus_feature.csv               # 버스 관련 데이터
│   ├── subway_feature.csv            # 지하철 관련 데이터
│   └── sample_submission.csv         # 제출 양식
│
├── [개발자명]/                        # 개발자별 작업 디렉토리 (예: sangwon, sanghyeon, jihoon)
│   ├── random-forest/                # Random Forest 모델
│   │   ├── requirements.txt          # RandomForest 모델 의존성
│   │   ├── main.py                   # 메인 스크립트
│   │   ├── ...                       # 기타 모듈 파일
│   │   └── venv/                     # 가상환경 디렉토리 (git에서 제외됨)
│   │
│   ├── xgboost/                      # XGBoost 모델 (예시)
│   │   ├── requirements.txt          # XGBoost 모델 의존성
│   │   └── ...
│   │
│   └── neural-network/               # 신경망 모델 (예시)
│       ├── requirements.txt          # 신경망 모델 의존성
│       └── ...
│
├── model-template/                   # 모델 템플릿 디렉토리
├── setup_env.py                      # 가상환경 설정 유틸리티
├── check_venv.py                     # 가상환경 확인 스크립트
├── config.ini                        # 개발자 설정 파일 (git에서 제외됨)
└── README.md                         # 프로젝트 설명 문서
```

## 프로젝트 시작하기

### 1단계: 처음 설정 (최초 1회만 실행)

1. **개발자 이름 설정**

```bash
# 개발자 이름 설정 (예: sangwon)
python setup_env.py set_developer_name sangwon
```

2. **개발자 디렉토리 생성**

```bash
# 개발자 이름으로 디렉토리 생성
mkdir sangwon
```

3. **모델 디렉토리 생성**

```bash
# 모델 템플릿을 복사하여 새 모델 디렉토리 생성
cp -r model-template sangwon/random-forest
```

4. **가상환경 생성**

```bash
# 모델 디렉토리로 이동
cd sangwon/random-forest

# 가상환경 생성
python -m venv venv
```

5. **가상환경 활성화 및 패키지 설치**

Windows:

```bash
# 가상환경 활성화
venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 설치된 패키지 확인
pip list
```

macOS/Linux:

```bash
# 가상환경 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 설치된 패키지 확인
pip list
```

6. **가상환경 비활성화**

```bash
# 작업 완료 후 가상환경 비활성화
deactivate
```

### 2단계: 일상적인 개발 작업 시

1. **가상환경 활성화**

```bash
# 모델 디렉토리로 이동
cd sangwon/random-forest

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 또는 (macOS/Linux)
source venv/bin/activate
```

2. **코드 작성 및 실행**

```bash
# 모델 실행
python main.py

# 또는 특정 모듈 테스트
python data_loader.py
```

3. **패키지 추가 설치 시**

```bash
# 새 패키지 설치
pip install pandas

# requirements.txt 업데이트
pip freeze > requirements.txt
```

4. **가상환경 비활성화**

```bash
# 작업 완료 후 가상환경 비활성화
deactivate
```

### 3단계: 새 모델 추가 시

1. **새 모델 디렉토리 생성**

```bash
# 자신의 개발자 디렉토리에 새 모델 디렉토리 생성
cp -r model-template sangwon/xgboost
```

2. **새 모델의 가상환경 설정**

```bash
# 새 모델 디렉토리로 이동
cd sangwon/xgboost

# 가상환경 생성 및 활성화 (이하 동일)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## 유틸리티 도구 사용법

### 개발자 이름 설정

개발자 이름은 `config.ini` 파일에 저장되며, 이 파일은 git에서 관리되지 않습니다.

```bash
# 개발자 이름 설정
python setup_env.py set_developer_name [이름]

# 현재 설정 확인
python setup_env.py config
```

### 가상환경 확인

```bash
# 현재 활성화된 가상환경 정보 확인
python check_venv.py
```

### 모델 목록 확인

```bash
# 개발자 디렉토리의 모델 목록 확인
python setup_env.py list
```

## 파일 및 디렉토리 설명

- `setup_env.py`: 가상환경 관리 지원 유틸리티
- `check_venv.py`: 현재 활성화된 가상환경 정보 확인 도구
- `config.ini`: 개발자 설정 파일 (git에서 제외됨)
- `model-template`: 새 모델 생성을 위한 템플릿 디렉토리

각 모델 디렉토리는 다음과 같은 파일들로 구성됩니다:

- `main.py`: 메인 스크립트
- `data_loader.py`: 데이터 로딩 모듈
- `preprocessor.py`: 데이터 전처리 모듈
- `model_trainer.py`: 모델 훈련 모듈
- `evaluator.py`: 모델 평가 모듈
- `utils.py`: 유틸리티 함수 모듈
- `requirements.txt`: 의존성 패키지 목록
- `venv/`: 가상환경 디렉토리 (git에서 제외됨)
- `output/`: 결과 저장 디렉토리 (git에서 제외됨)
- `models/`: 학습된 모델 저장 디렉토리

## 경로 참조 방식

각 모델의 스크립트는 모델 디렉토리 내에서 실행되는 것을 가정하고 작성되었습니다:

1. **데이터 로드 경로**:

   - 원본 데이터는 상대 경로 `../../raw/`로 참조됩니다.
   - 예: `../../raw/train.csv`, `../../raw/test.csv` 등

2. **모델 및 결과 저장 경로**:
   - 모델은 모델 디렉토리 내의 `models/` 폴더에 저장됩니다.
   - 결과는 모델 디렉토리 내의 `output/` 폴더에 저장됩니다.
   - 예: `output/feature_importance.csv`, `models/random_forest_model.pkl` 등

따라서 모델 스크립트를 실행할 때는 해당 모델 디렉토리에서 실행해야 합니다:

```bash
# 모델 디렉토리로 이동
cd sangwon/random-forest

# 가상환경 활성화
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 모델 실행
python main.py
```

이렇게 하면 각 모델은 독립적인 환경에서 실행되며, 각자의 출력 및 모델 파일을 자신의 디렉토리에 저장합니다.

## 협업 주의사항

1. **가상환경은 공유하지 않습니다**

   - 가상환경 디렉토리(`venv/`)는 `.gitignore`에 추가되어 있어 저장소에 포함되지 않습니다.
   - 각 개발자는 자신의 로컬 환경에서 가상환경을 직접 생성해야 합니다.

2. **requirements.txt는 최신 상태로 유지합니다**

   - 새 패키지를 설치하면 `pip freeze > requirements.txt` 명령으로 의존성 목록을 업데이트합니다.
   - 다른 개발자가 동일한 환경을 재현할 수 있도록 정확한 버전을 명시합니다.

3. **config.ini는 공유하지 않습니다**

   - 개발자 이름 설정이 포함된 `config.ini` 파일은 `.gitignore`에 추가되어 있어 저장소에 포함되지 않습니다.
   - 각 개발자는 처음 프로젝트를 시작할 때 `setup_env.py set_developer_name` 명령으로 자신의 설정을 해야 합니다.

4. **다른 개발자의 디렉토리는 참고만 합니다**
   - 다른 개발자의 디렉토리는 참고 목적으로만 사용하고, 직접 수정하지 않습니다.
   - 공통 코드는 별도의 공유 디렉토리에 배치하는 것이 좋습니다.

---
