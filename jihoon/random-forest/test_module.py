#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
부동산 가격 예측 프로젝트 - Random Forest 모델
테스트 모듈

이 모듈은 프로젝트의 모든 모듈이 올바르게 동작하는지 테스트합니다.
"""

import os
import sys
import time
import importlib.util

def check_module(module_name):
    """
    모듈이 존재하는지 확인하고 임포트 가능한지 테스트하는 함수
    
    Args:
        module_name (str): 테스트할 모듈 이름
        
    Returns:
        bool: 모듈이 존재하고 임포트 가능하면 True, 아니면 False
    """
    try:
        # 모듈 파일 경로
        module_path = f'taem/random-forest/{module_name}.py'
        
        # 모듈 파일이 존재하는지 확인
        if not os.path.exists(module_path):
            print(f"❌ 모듈 파일이 존재하지 않습니다: {module_path}")
            return False
        
        # 모듈 임포트 시도
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"✅ 모듈 '{module_name}' 임포트 성공!")
        return True
    
    except Exception as e:
        print(f"❌ 모듈 '{module_name}' 임포트 실패: {str(e)}")
        return False

def check_dependencies():
    """
    필요한 라이브러리가 설치되어 있는지 확인하는 함수
    
    Returns:
        bool: 모든 라이브러리가 설치되어 있으면 True, 아니면 False
    """
    libraries = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'joblib', 'statsmodels'
    ]
    
    all_installed = True
    
    print("필요한 라이브러리 설치 상태 확인:")
    for lib in libraries:
        try:
            importlib.import_module(lib)
            print(f"✅ {lib} 설치됨")
        except ImportError:
            print(f"❌ {lib} 설치되지 않음")
            all_installed = False
    
    return all_installed

def main():
    """
    메인 테스트 함수
    """
    print("=" * 60)
    print("부동산 가격 예측 프로젝트 - Random Forest 모델 테스트")
    print("=" * 60)
    
    # 의존성 확인
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n⚠️ 일부 라이브러리가 설치되지 않았습니다.")
        print("   'pip install -r taem/random-forest/requirements.txt' 명령으로 설치해주세요.")
    
    print("\n모듈 테스트 시작:")
    # 핵심 모듈 확인
    modules_to_check = [
        'main', 'data_loader', 'preprocessor', 
        'model_trainer', 'evaluator', 'utils'
    ]
    
    all_modules_ok = True
    for module in modules_to_check:
        if not check_module(module):
            all_modules_ok = False
    
    # 전체 결과 출력
    print("\n테스트 결과 요약:")
    if deps_ok and all_modules_ok:
        print("✅ 모든 테스트가 통과되었습니다!")
        print("   'python taem/random-forest/main.py' 명령으로 프로젝트를 실행할 수 있습니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        if not deps_ok:
            print("   - 필요한 라이브러리를 설치해주세요.")
        if not all_modules_ok:
            print("   - 모듈 오류를 수정해주세요.")

if __name__ == "__main__":
    main() 