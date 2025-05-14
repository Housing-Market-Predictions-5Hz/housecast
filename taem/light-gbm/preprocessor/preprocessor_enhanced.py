import pandas as pd
import numpy as np
from preprocessor.column_tags import TAGS
from pathlib import Path
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings("ignore")

CURRENT_YEAR = 2023
GANGNAM_CENTER_X = 203731
GANGNAM_CENTER_Y = 452331
# 파일 경로 수정
DONG_LATLONG_PATH = Path("../../raw/latlng.bdong.txt")
APT_COORDS_PATH = Path("../../raw/apt_all.csv")

# 외부 단지코드 불러오기
def load_complex_code(path="../../raw/Complex Code_20221129.xlsx"):
    df = pd.read_excel(path)
    df["매칭키"] = df["시군구"].astype(str) + "_" + df["건물명"].astype(str)
    return df[["매칭키", "단지코드"]].dropna()

# 아파트 좌표 정보 불러오기
def load_apt_coordinates():
    """
    apt_all.csv 파일에서 아파트 주소, 위도, 경도 정보를 불러옵니다.
    
    Returns:
        pd.DataFrame: 아파트 주소 및 좌표 정보가 포함된 데이터프레임
    """
    try:
        if not APT_COORDS_PATH.exists():
            print(f"  - 아파트 좌표 파일({APT_COORDS_PATH})을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # CSV 파일 불러오기
        df = pd.read_csv(APT_COORDS_PATH)
        
        # 칼럼 이름 확인
        print(f"  - apt_all.csv 칼럼: {df.columns.tolist()}")
        
        # 필요한 컬럼만 사용
        if 'Address' in df.columns and 'Latitude' in df.columns and 'Longitude' in df.columns:
            # 데이터 타입 변환
            df['Address'] = df['Address'].astype(str)
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            
            # 결측치 제거
            df = df.dropna(subset=['Address'])
            
            # 주소에서 시군구와 아파트명 추출
            def extract_sigungu_and_apt(address):
                try:
                    parts = address.split()
                    if len(parts) < 3:
                        return "", ""
                    
                    # 시군구: 첫 두 단어 (예: "서울특별시 강남구")
                    sigungu = " ".join(parts[:2])
                    
                    # 아파트명: 주소의 마지막 부분에 있는 아파트 이름 (예: "개포우성3차")
                    # 일반적으로 마지막 단어가 아파트명인 경우가 많음
                    apt_name = parts[-1]
                    
                    return sigungu, apt_name
                except:
                    return "", ""
            
            # 시군구와 아파트명 추출
            df['시군구'], df['아파트명'] = zip(*df['Address'].apply(extract_sigungu_and_apt))
            
            # 좌표 정보 설정
            df['좌표X'] = df['Longitude']
            df['좌표Y'] = df['Latitude']
            
            # 필요한 컬럼만 선택
            result = df[['시군구', '아파트명', 'Address', '좌표X', '좌표Y']].copy()
            # Address 컬럼명 변경
            result = result.rename(columns={'Address': '주소'})
            # 결측치 제거
            result = result.dropna(subset=['좌표X', '좌표Y'])
            
            print(f"  - 아파트 좌표 정보 {len(result)}개를 불러왔습니다.")
            return result
        else:
            print(f"  - 아파트 좌표 파일의 형식이 올바르지 않습니다. 필요한 컬럼: Address, Latitude, Longitude")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"  - 아파트 좌표 정보 불러오기 중 오류: {e}")
        return pd.DataFrame()

# 법정동 중심 좌표 불러오기
def load_dong_center_coordinates():
    """
    법정동 중심 좌표(위도, 경도)를 불러와 좌표X, 좌표Y로 반환
    
    Returns:
        pd.DataFrame: 법정동, 좌표X_c, 좌표Y_c 컬럼을 가진, 법정동 좌표 정보 DataFrame
    """
    try:
        if not DONG_LATLONG_PATH.exists():
            print(f"  - 법정동 좌표 파일({DONG_LATLONG_PATH})을 찾을 수 없습니다.")
            return pd.DataFrame(columns=["동", "좌표X_c", "좌표Y_c"])
        
        # 파일 불러오기 (탭으로 구분된 텍스트 파일)
        columns = ["법정동코드", "코드2", "시도", "구", "동", "empty", "좌표"]
        df = pd.read_csv(DONG_LATLONG_PATH, sep="\t", header=None, names=columns, encoding="utf-8")
        
        # 좌표 정보 분리 (위도,경도 형식)
        df[["위도", "경도"]] = df["좌표"].str.split(",", expand=True)
        
        # 숫자로 변환
        df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
        df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
        
        # 좌표계 변환 (위도/경도를 좌표X/좌표Y로 변환) - WGS84를 UTM-K로 단순 변환
        df["좌표X_c"] = df["경도"] * 10000  # 간단한 변환
        df["좌표Y_c"] = df["위도"] * 10000  # 간단한 변환
        
        # 필요한 컬럼만 선택하여 반환
        result_df = df[["동", "좌표X_c", "좌표Y_c"]].dropna()
        
        print(f"  - 법정동 좌표 정보 {len(result_df)}개를 불러왔습니다.")
        return result_df
        
    except Exception as e:
        print(f"  - 법정동 좌표 정보 불러오기 중 오류: {e}")
        return pd.DataFrame(columns=["동", "좌표X_c", "좌표Y_c"])

# 전용면적 구간화
def bin_area(x):
    if x < 60: return '소형'
    elif x < 85: return '중형'
    elif x < 135: return '대형'
    else: return '초대형'

# 층수 구간화
def bin_floor(x):
    if pd.isna(x): return '미상'
    if x <= 5: return '저층'
    elif x <= 15: return '중층'
    else: return '고층'

def load_data(train_path, test_path, bus_path, subway_path, submission_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    bus = pd.read_csv(bus_path)
    subway = pd.read_csv(subway_path)
    sample_submission = pd.read_csv(submission_path)
    return train, test, bus, subway, sample_submission

# 좌표 관련 컬럼 이름 찾기
def coord_cols(df):
    """좌표 컬럼 이름 찾기"""
    cand_lon = ["좌표X", "경도", "longitude", "lon", "WGS84_X", "X좌표", "x"]
    cand_lat = ["좌표Y", "위도", "latitude", "lat", "WGS84_Y", "Y좌표", "y"]
    lon = next((c for c in cand_lon if c in df.columns), None)
    lat = next((c for c in cand_lat if c in df.columns), None)
    if lon is None or lat is None:
        raise ValueError("Longitude / latitude columns not found!")
    return lon, lat

# 교통 정보용 BallTree 구축
def build_ball_tree(df):
    """좌표 기반 BallTree 생성"""
    lon, lat = coord_cols(df)
    return BallTree(np.deg2rad(df[[lon, lat]].values), metric="haversine")

# 교통 정보 추가
def add_transport_features(df, tree_sub, tree_bus):
    """교통 관련 특성 추가"""
    lon, lat = coord_cols(df)
    
    # NaN이 있는 행 처리
    mask_valid = ~(df[lon].isna() | df[lat].isna())
    
    if not mask_valid.all():
        print(f"  - 좌표 NaN 값이 {(~mask_valid).sum()}개 행에서 발견되었습니다.")
        raise ValueError("좌표에 NaN 값이 포함되어 있습니다. 좌표 결측치를 보간 후 실행하세요.")
    
    # 좌표 변환
    coords_rad = np.deg2rad(df[[lon, lat]].values)
    R = 6_371_000  # 지구 반경 (미터)
    
    # 지하철 관련 특성
    d_sub, _ = tree_sub.query(coords_rad, k=1)
    df["dist_subway_min"] = d_sub[:, 0] * R
    df["cnt_subway_500"] = tree_sub.query_radius(coords_rad, r=500 / R, count_only=True)
    df["cnt_subway_1000"] = tree_sub.query_radius(coords_rad, r=1000 / R, count_only=True)
    
    # 버스 관련 특성
    d_bus, _ = tree_bus.query(coords_rad, k=1)
    df["dist_bus_min"] = d_bus[:, 0] * R
    df["cnt_bus_300"] = tree_bus.query_radius(coords_rad, r=300 / R, count_only=True)
    
    # 누락되는 컬럼이 없는지 확인
    required_cols = ['dist_subway_min', 'cnt_subway_500', 'cnt_subway_1000', 'dist_bus_min', 'cnt_bus_300']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"교통 데이터 컬럼이 누락되었습니다: {missing_cols}")
    
    return df

# 시군구 컬럼에서 동 추출 함수
def extract_dong_from_sigungu(sigungu):
    """시군구 컬럼에서 동 정보 추출"""
    try:
        parts = sigungu.split()
        if len(parts) >= 3:
            return parts[2]  # 일반적으로 '서울특별시 강남구 삼성동' 형태에서 동 추출
        return ""
    except:
        return ""

def boost_coordinates(combined):
    """
    좌표 결측치 보강 함수 - 향상된 버전
    1. 아파트 주소, 이름 기반 좌표 매칭 (apt_all.csv 활용)
    2. 아파트명 기준 forward/backward fill
    3. 법정동 중심 좌표로 채우기
    4. 남은 결측치는 중앙값으로 채우기
    """
    print("좌표 결측치 보강 중...")
    
    # 결측치 상태 확인
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            missing_before = combined[col].isna().sum()
            missing_pct = missing_before / len(combined) * 100
            print(f"  - {col} 결측치: {missing_before}개 ({missing_pct:.2f}%)")
    
    # 1. apt_all.csv 데이터를 활용한 아파트 이름, 주소 기반 좌표 매칭
    print("  - 아파트 주소, 이름 기반 좌표 매칭 중...")
    apt_coords = load_apt_coordinates()
    
    # 시군구 정규화 함수 (테스트 데이터는 "서울특별시 강서구 염창동" 형식, apt_all.csv는 "서울특별시 강남구" 형식)
    def normalize_sigungu(sigungu):
        if pd.isna(sigungu):
            return ""
        parts = str(sigungu).split()
        if len(parts) >= 2:
            return " ".join(parts[:2])  # 첫 두 부분만 사용 (시/도 + 구)
        return sigungu
    
    # 데이터프레임에 정규화된 시군구 추가
    combined['시군구_정규화'] = combined['시군구'].apply(normalize_sigungu)
    
    print(f"  - 시군구 정규화 예시: {combined['시군구'].iloc[0]} -> {combined['시군구_정규화'].iloc[0]}")
    
    if not apt_coords.empty and '시군구' in combined.columns and '아파트명' in combined.columns:
        # 아파트명 정규화 함수 (특수문자 제거, 소문자 변환 등)
        def normalize_apt_name(name):
            if pd.isna(name):
                return ""
            # 소문자 변환, 특수문자 및 공백 제거
            import re
            name = str(name).lower()
            # '아파트', '단지' 등 공통 접미사 제거
            name = re.sub(r'아파트$', '', name)
            # 숫자만 남기기 (예: "1단지" -> "1")
            if re.search(r'\d+단지$', name):
                name = re.sub(r'단지$', '', name)
            # 특수문자 제거
            name = re.sub(r'[^\w\s]', '', name)
            # 공백 제거
            name = re.sub(r'\s+', '', name)
            return name

        # 데이터프레임에 정규화된 아파트명 추가
        combined['아파트명_정규화'] = combined['아파트명'].apply(normalize_apt_name)
        apt_coords['아파트명_정규화'] = apt_coords['아파트명'].apply(normalize_apt_name)
        
        # 1-1. 시군구와 정규화된 아파트명 완전 일치하는 경우 먼저 매칭
        print("  - 완전 일치 매칭 시도 중...")
        
        # 결측치가 있는 행만 필터링
        missing_coords = combined[combined['좌표X'].isna() | combined['좌표Y'].isna()].copy()
        print(f"  - 매칭 대상 행: {len(missing_coords)}개")
        
        # 매칭 시도 (효율성을 위해 벡터화)
        match_count = 0
        
        # 효율적인 매칭을 위해 apt_coords를 딕셔너리로 변환
        apt_coords_dict = {}
        for _, row in apt_coords.iterrows():
            key = (row['시군구'], row['아파트명_정규화'])
            apt_coords_dict[key] = (row['좌표X'], row['좌표Y'])
        
        # 완전 일치 매칭 (시군구_정규화 + 아파트명_정규화)
        for idx, row in missing_coords.iterrows():
            sigungu = row['시군구_정규화']
            apt_name_norm = row['아파트명_정규화']
            
            if pd.notna(sigungu) and pd.notna(apt_name_norm) and apt_name_norm.strip() != '':
                key = (sigungu, apt_name_norm)
                if key in apt_coords_dict:
                    combined.loc[idx, '좌표X'] = apt_coords_dict[key][0]
                    combined.loc[idx, '좌표Y'] = apt_coords_dict[key][1]
                    match_count += 1
        
        print(f"  - 완전 일치 매칭으로 {match_count}개 좌표를 찾았습니다.")
        
        # 1-2. 결측치가 남아있다면 부분 매칭 시도
        missing_mask = combined["좌표X"].isna() | combined["좌표Y"].isna()
        if missing_mask.any():
            print("  - 부분 매칭 시도 중...")
            partial_match_count = 0
            
            # 결측치가 있는 행만 다시 필터링
            missing_coords = combined[missing_mask].copy()
            print(f"  - 부분 매칭 대상 행: {len(missing_coords)}개")
            
            # 각 시군구별로 apt_coords를 미리 필터링하여 성능 향상
            sigungu_groups = {}
            for sg in apt_coords['시군구'].unique():
                sigungu_groups[sg] = apt_coords[apt_coords['시군구'] == sg]
            
            # 부분 매칭 시도
            for idx, row in missing_coords.iterrows():
                sigungu = row['시군구_정규화']
                apt_name_norm = row['아파트명_정규화']
                
                if pd.notna(sigungu) and pd.notna(apt_name_norm) and apt_name_norm.strip() != '':
                    # 해당 시군구 그룹 가져오기
                    sg_group = sigungu_groups.get(sigungu, pd.DataFrame())
                    
                    # 매칭 시도
                    matches = pd.DataFrame()
                    
                    # 1. 아파트명 포함 관계 확인
                    if not sg_group.empty:
                        # 방법 1: 데이터의 아파트명이 apt_coords의 아파트명에 포함
                        matches = sg_group[sg_group['아파트명_정규화'].str.contains(apt_name_norm, na=False)]
                        
                        # 2. apt_coords의 아파트명이 데이터의 아파트명에 포함
                        if matches.empty and len(apt_name_norm) > 3:
                            for _, sg_row in sg_group.iterrows():
                                if pd.notna(sg_row['아파트명_정규화']) and (
                                    sg_row['아파트명_정규화'] in apt_name_norm or 
                                    apt_name_norm in sg_row['아파트명_정규화']):
                                    matches = pd.DataFrame([sg_row])
                                    break
                    
                    # 3. 브랜드명으로 매칭 (예: "래미안", "힐스테이트" 등)
                    if matches.empty and len(apt_name_norm) > 2 and not sg_group.empty:
                        brand_name = apt_name_norm[:2]  # 처음 2글자로 시도
                        matches = sg_group[sg_group['아파트명_정규화'].str.startswith(brand_name, na=False)]
                    
                    if not matches.empty:
                        combined.loc[idx, '좌표X'] = matches.iloc[0]['좌표X']
                        combined.loc[idx, '좌표Y'] = matches.iloc[0]['좌표Y']
                        partial_match_count += 1
            
            print(f"  - 부분 매칭으로 {partial_match_count}개 좌표를 찾았습니다.")
        
        # 임시 컬럼 제거
        temp_cols = ['아파트명_정규화', '시군구_정규화']
        for col in temp_cols:
            if col in combined.columns:
                combined.drop(columns=[col], inplace=True)
    
    # 결측치 상태 확인
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            missing_after_apt = combined[col].isna().sum()
            missing_pct = missing_after_apt / len(combined) * 100
            print(f"  - 아파트 좌표 매칭 후 {col} 결측치: {missing_after_apt}개 ({missing_pct:.2f}%)")
    
    # 2. 시군구+아파트명 그룹에 대해 ffill/bfill (Index 고유성 보장을 위해 시군구 추가)
    print("  - 아파트별 좌표 보간 중...")
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            # 그룹별로 결측치 채우기
            combined[col + "_filled"] = combined[col]
            grouped = combined.groupby(["시군구", "아파트명"])
            group_median = grouped[col].transform(lambda x: x.median())
            combined[col + "_filled"] = combined[col].fillna(group_median)
            
            # 다른 아파트와 구분되는 고유 식별자 생성
            combined["apt_id"] = combined["시군구"].astype(str) + "_" + combined["아파트명"].astype(str)
            
            # 기존 값을 새 컬럼으로 대체
            combined[col] = combined[col + "_filled"]
            combined.drop(columns=[col + "_filled"], inplace=True)
    
    # 결측치 상태 중간 확인
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            missing_mid = combined[col].isna().sum()
            missing_pct = missing_mid / len(combined) * 100
            print(f"  - 아파트명 기준 보간 후 {col} 결측치: {missing_mid}개 ({missing_pct:.2f}%)")
    
    # 3. 법정동 중심 좌표로 채우기
    print("  - 법정동 중심 좌표로 채우는 중...")
    dong_center = load_dong_center_coordinates()
    
    if not dong_center.empty and '시군구' in combined.columns:
        # 시군구 컬럼에서 동 정보 추출
        if '동' not in combined.columns:
            print("  - '법정동' 컬럼이 없어서 시군구에서 동 정보를 추출합니다")
            combined['동'] = combined['시군구'].fillna('').apply(extract_dong_from_sigungu)
            
        # 동 기준으로 좌표 보간
        merged = combined.merge(dong_center, on='동', how='left')
        
        # merge 결과 체크
        matched_count = merged['좌표X_c'].notna().sum()
        print(f"  - 동 정보 매칭 결과: {matched_count}/{len(merged)}개 매칭됨")
        
        # 추출된 좌표로 결측치 채우기
        for col, c_col in zip(['좌표X', '좌표Y'], ['좌표X_c', '좌표Y_c']):
            if col in combined.columns and c_col in merged.columns:
                combined[col] = combined[col].fillna(merged[c_col])
        
        # 결측치 상태 다시 확인
        for col in ["좌표X", "좌표Y"]:
            if col in combined.columns:
                missing_after_dong = combined[col].isna().sum()
                missing_pct = missing_after_dong / len(combined) * 100
                print(f"  - 법정동 중심 좌표 보간 후 {col} 결측치: {missing_after_dong}개 ({missing_pct:.2f}%)")
    else:
        print("  - 법정동 중심 좌표 정보를 사용할 수 없습니다.")
    
    # 4. 남은 결측치는 중앙값으로 채우기
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            col_median = combined[col].median()
            combined[col] = combined[col].fillna(col_median)
    
    # 최종 결측치 확인
    for col in ["좌표X", "좌표Y"]:
        if col in combined.columns:
            missing_after = combined[col].isna().sum()
            if missing_after > 0:
                print(f"  - 경고: {col}에 여전히 {missing_after}개의 결측치가 있습니다.")
            else:
                print(f"  - {col} 결측치 보강 완료!")
    
    # 임시로 사용한 컬럼 제거
    temp_cols = ["apt_id", "동"]
    for col in temp_cols:
        if col in combined.columns:
            combined.drop(columns=[col], inplace=True)
    
    return combined

# 지역 정보 추출 (district, gangnam7)
def extract_district_features(df):
    """시군구에서 구 정보를 추출하고 강남7 지역 여부 표시"""
    print("  - 지역 정보 추출 중...")
    
    # 시군구 컬럼이 없으면 건너뛰기
    if "시군구" not in df.columns:
        print("    - '시군구' 컬럼이 없어 지역 정보 추출을 건너뜁니다.")
        return df
    
    # 시군구 컬럼 타입 변환 (문자열이 아니면 변환)
    if not pd.api.types.is_string_dtype(df["시군구"]):
        print("    - '시군구' 컬럼을 문자열 타입으로 변환합니다.")
        df["시군구"] = df["시군구"].astype(str)
    
    # NaN이나 빈 문자열 있는지 확인
    if df["시군구"].isna().any() or (df["시군구"] == "").any():
        print("    - '시군구' 컬럼에 결측치나 빈 문자열이 있어 대체합니다.")
        df["시군구"] = df["시군구"].fillna("미상_미상_미상")
        df.loc[df["시군구"] == "", "시군구"] = "미상_미상_미상"
    
    # 시군구에서 구 추출 함수
    def extract_gu(sigungu):
        try:
            parts = sigungu.split()
            if len(parts) >= 2:
                return parts[1]  # 두 번째 부분이 '구'
            return "미상"
        except:
            return "미상"
    
    # 법정구 추출 및 강남7 지역 여부 표시
    try:
        df["district"] = df["시군구"].apply(extract_gu)
        
        # 강남7 지역 여부
        GANGNAM7 = {"강서구", "영등포구", "동작구", "서초구", "강남구", "송파구", "강동구"}
        df["is_gangnam7"] = df["district"].isin(GANGNAM7).astype(int)
        
        print("    - 지역 정보 추출 완료!")
    except Exception as e:
        print(f"    - 지역 정보 추출 중 오류 발생: {e}")
        print("    - 기본값으로 대체합니다.")
        df["district"] = "미상"
        df["is_gangnam7"] = 0
    
    return df

def preprocess_data(train, test, bus, subway):
    print("데이터 전처리 시작...")
    
    train["is_train"] = 1
    test["is_train"] = 0
    test["target"] = np.nan
    combined = pd.concat([train, test], axis=0)

    # ✅ 외부 단지코드 병합
    try:
        complex_map = load_complex_code()
        combined["매칭키"] = combined["시군구"].astype(str) + "_" + combined["아파트명"].astype(str)
        combined = pd.merge(combined, complex_map, how="left", on="매칭키")
    except Exception as e:
        print(f"단지코드 매핑 중 오류: {e}")
        print("단지코드 매핑을 건너뜁니다.")

    # ✅ 단지코드 존재 확인
    if "단지코드" not in combined.columns:
        print("'단지코드' 컬럼이 없어 자체 코드를 생성합니다.")
        # 아파트명과 시군구로 단지코드 대체 생성
        combined["단지코드"] = combined["시군구"].astype(str) + "_" + combined["아파트명"].astype(str)
        combined["단지코드"] = pd.factorize(combined["단지코드"])[0]
    
    # ✅ 좌표 결측치 보강 (enhanced 방식)
    combined = boost_coordinates(combined)

    # ✅ 컬럼 태그 기반 처리
    print("컬럼 태그 기반 처리 중...")
    for col, tag in TAGS.items():
        if col not in combined.columns:
            continue
        if tag == "drop":
            combined.drop(columns=col, inplace=True)
        elif tag == "flag":
            combined[f"is_na_{col}"] = combined[col].isnull().astype(int)
            combined.drop(columns=col, inplace=True)
        elif tag == "impute":
            if combined[col].dtype == "object":
                combined[col].fillna("미상", inplace=True)
                combined[col] = combined[col].astype("category").cat.codes
            else:
                combined[col].fillna(combined[col].median(), inplace=True)
        elif tag == "coord":
            # 앞에서 이미 좌표 보강을 했기 때문에 여기서는 최종 확인만 수행
            combined[col].fillna(0, inplace=True)
        elif tag == "categorical":
            combined[col] = combined[col].fillna("미상")
            combined[col] = combined[col].astype("category").cat.codes
        elif tag == "keep":
            combined[col].fillna("미상" if combined[col].dtype == "object" else combined[col].median(), inplace=True)

    # ✅ 특성 엔지니어링
    print("특성 엔지니어링 중...")
    if "건축년도" in combined.columns:
        combined["building_age"] = CURRENT_YEAR - combined["건축년도"]
        combined["building_age"] = combined["building_age"].clip(lower=0)

    if "전용면적(㎡)" in combined.columns:
        combined["전용면적_bin"] = combined["전용면적(㎡)"].apply(bin_area)
        combined["전용면적_bin"] = combined["전용면적_bin"].astype("category").cat.codes

    if "층" in combined.columns:
        combined["층수_bin"] = combined["층"].apply(bin_floor)
        combined["층수_bin"] = combined["층수_bin"].astype("category").cat.codes
        if "전용면적(㎡)" in combined.columns:
            combined["floor_x_area"] = combined["층"] * combined["전용면적(㎡)"]

    if "좌표X" in combined.columns and "좌표Y" in combined.columns:
        combined["distance_from_gangnam_center"] = np.sqrt(
            (combined["좌표X"] - GANGNAM_CENTER_X) ** 2 + (combined["좌표Y"] - GANGNAM_CENTER_Y) ** 2
        )

    if "시군구" in combined.columns and "단지코드" in combined.columns:
        combined["시군구_단지"] = combined["시군구"].astype(str) + "_" + combined["단지코드"].astype(str)
        combined["시군구_단지"] = combined["시군구_단지"].astype("category").cat.codes

    if "target" in combined.columns:
        mean_target_by_apt = combined[combined['is_train'] == 1].groupby("단지코드")["target"].mean()
        combined["단지별_평균가"] = combined["단지코드"].map(mean_target_by_apt)
        combined["단지별_평균가"].fillna(combined["단지별_평균가"].median(), inplace=True)
    
    # ✅ 지역 관련 특성 추가 (district, is_gangnam7)
    combined = extract_district_features(combined)
    
    # ✅ 교통 데이터 처리 (BallTree 활용)
    print("교통 데이터 처리 중...")
    # BallTree 구축
    tree_bus = build_ball_tree(bus)
    tree_subway = build_ball_tree(subway)
    
    # 교통 특성 추가 - 예외 처리 없이 직접 실행
    combined = add_transport_features(combined, tree_subway, tree_bus)
    print("  - BallTree 기반 교통 특성 추가 완료!")

    # ✅ 데이터 분할
    print("데이터 분할 처리 중...")
    train_processed = combined[combined["is_train"] == 1].drop(columns=["is_train"])
    test_processed = combined[combined["is_train"] == 0].drop(columns=["is_train", "target"])

    print(f"전처리 완료: train_processed 형태 {train_processed.shape}, test_processed 형태 {test_processed.shape}")
    return train_processed, test_processed, bus, subway

if __name__ == "__main__":
    # 테스트 실행
    from pathlib import Path
    
    data_dir = Path("../../raw")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    bus_path = data_dir / "bus_feature.csv"
    subway_path = data_dir / "subway_feature.csv"
    submission_path = data_dir / "sample_submission.csv"
    
    train, test, bus, subway, sample_submission = load_data(
        train_path, test_path, bus_path, subway_path, submission_path
    )
    
    train_processed, test_processed, _, _ = preprocess_data(train, test, bus, subway)
    
    print("전처리 완료!")
    print(f"훈련 데이터 크기: {train_processed.shape}")
    print(f"테스트 데이터 크기: {test_processed.shape}")
    print("\n훈련 데이터 컬럼:")
    print(train_processed.columns.tolist())
