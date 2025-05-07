#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
현재 Python 환경이 가상환경인지 확인하는 스크립트
"""

import os
import sys
import site
import platform
import configparser

# config.ini 파일에서 개발자 이름 설정을 읽어옴
def get_developer_name():
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    
    # 기본 개발자 이름
    default_name = 'developer'
    
    # config.ini 파일이 존재하지 않으면 생성
    if not os.path.exists(config_path):
        print(f"config.ini 파일이 존재하지 않습니다.")
        print(f"기본 개발자 이름을 사용합니다: {default_name}")
        return default_name
    
    # config.ini 파일에서 개발자 이름 읽기
    try:
        config.read(config_path)
        developer_name = config.get('developer', 'name', fallback=default_name)
        return developer_name
    except Exception as e:
        print(f"설정 파일 읽기 오류: {e}")
        print(f"기본 개발자 이름을 사용합니다: {default_name}")
        return default_name

# 개발자 이름 가져오기
DEVELOPER_NAME = get_developer_name()

def is_virtual_env():
    """가상환경인지 확인합니다."""
    # 환경 변수로 확인
    if hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix:
        return True
    
    # VIRTUAL_ENV 환경변수로 확인
    if 'VIRTUAL_ENV' in os.environ:
        return True
    
    return False

def main():
    """메인 함수"""
    print("\n===== Python 환경 정보 =====")
    print(f"개발자 이름: {DEVELOPER_NAME}")
    print(f"Python 실행 경로: {sys.executable}")
    print(f"Python 버전: {sys.version}")
    print(f"sys.prefix: {sys.prefix}")
    
    if hasattr(sys, 'real_prefix'):
        print(f"sys.real_prefix: {sys.real_prefix}")
    
    if hasattr(sys, 'base_prefix'):
        print(f"sys.base_prefix: {sys.base_prefix}")
    
    print(f"site-packages: {site.getsitepackages()}")
    
    if 'VIRTUAL_ENV' in os.environ:
        print(f"VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}")
    
    print("\n===== 가상환경 여부 =====")
    if is_virtual_env():
        print("현재 가상환경이 활성화되어 있습니다.")
        
        # 어떤 모델의 가상환경인지 확인
        venv_path = os.environ.get('VIRTUAL_ENV', sys.prefix)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        developer_path = os.path.join(project_root, DEVELOPER_NAME)
        
        if os.path.exists(developer_path):
            found_model = False
            for model in os.listdir(developer_path):
                model_venv = os.path.join(developer_path, model, 'venv')
                if os.path.exists(model_venv):
                    try:
                        if os.path.samefile(venv_path, model_venv):
                            print(f"이 가상환경은 '{DEVELOPER_NAME}/{model}' 모델을 위한 환경입니다.")
                            found_model = True
                            break
                    except OSError:
                        # 경로가 존재하지만 samefile 비교에 실패한 경우 계속 진행
                        continue
            
            if not found_model:
                print(f"이 가상환경은 프로젝트의 모델 환경이 아닙니다.")
        else:
            print(f"'{DEVELOPER_NAME}' 개발자 디렉토리가 존재하지 않습니다.")
            print(f"setup_env.py set_developer_name [이름] 명령으로 개발자 이름을 설정하세요.")
    else:
        print("현재 가상환경이 활성화되어 있지 않습니다.")
        print("글로벌 Python 환경에서 실행 중입니다.")
        print("\n가상환경 활성화 방법:")
        if platform.system() == 'Windows':
            print(f"  {DEVELOPER_NAME}\\[모델명]\\venv\\Scripts\\activate")
        else:
            print(f"  source {DEVELOPER_NAME}/[모델명]/venv/bin/activate")

if __name__ == "__main__":
    main() 