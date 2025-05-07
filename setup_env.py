#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모델별 가상 환경 설정 스크립트

이 스크립트는 지정된 모델에 대한 가상 환경을 설정하고,
의존성 설치를 위한 명령어를 제공합니다.
"""

import os
import argparse
import subprocess
import sys
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
        config['developer'] = {'name': default_name}
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        print(f"config.ini 파일이 생성되었습니다. 기본 개발자 이름: {default_name}")
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

def set_developer_name(name):
    """
    개발자 이름을 설정합니다.
    
    Args:
        name (str): 설정할 개발자 이름
    """
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    
    # 기존 파일 읽기 또는 새 설정 생성
    if os.path.exists(config_path):
        config.read(config_path)
    
    if not config.has_section('developer'):
        config.add_section('developer')
    
    config.set('developer', 'name', name)
    
    # 파일에 저장
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"개발자 이름이 '{name}'으로 설정되었습니다.")
    print("이 설정은 config.ini 파일에 저장됩니다.")
    return True

def setup_environment(model_name):
    """
    지정된 모델에 대한 가상 환경을 설정합니다.
    
    Args:
        model_name (str): 모델 이름 (디렉토리 이름)
    """
    # 개발자 디렉토리 내의 모델 경로 설정
    model_path = os.path.join(DEVELOPER_NAME, model_name)
    
    # 모델 디렉토리 확인
    if not os.path.exists(model_path):
        print(f"오류: {model_path} 디렉토리가 존재하지 않습니다.")
        print(f"힌트: 'mkdir -p {model_path}' 명령어로 디렉토리를 생성하세요.")
        return False
    
    # requirements.txt 파일 확인
    req_path = os.path.join(model_path, 'requirements.txt')
    if not os.path.exists(req_path):
        print(f"오류: {req_path} 파일이 존재하지 않습니다.")
        print(f"힌트: model-template의 requirements.txt를 복사하거나 새로 생성하세요.")
        return False
    
    # 가상 환경 생성
    venv_path = os.path.join(model_path, 'venv')
    if not os.path.exists(venv_path):
        print(f"가상 환경 생성: {venv_path}")
        try:
            subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
        except subprocess.CalledProcessError:
            print("가상 환경 생성 실패. virtualenv를 사용해 보세요.")
            try:
                subprocess.run(['virtualenv', venv_path], check=True)
            except subprocess.CalledProcessError:
                print("virtualenv를 사용한 가상 환경 생성도 실패했습니다.")
                print("pip install virtualenv 명령어로 virtualenv를 설치하세요.")
                return False
    else:
        print(f"가상 환경이 이미 존재합니다: {venv_path}")
    
    # 의존성 설치 명령어 출력
    if platform.system() == 'Windows':  # Windows
        activate_cmd = f"{venv_path}\\Scripts\\activate"
        pip_cmd = f"{venv_path}\\Scripts\\pip"
    else:  # macOS/Linux
        activate_cmd = f"source {venv_path}/bin/activate"
        pip_cmd = f"{venv_path}/bin/pip"
    
    print("\n가상 환경 설정이 완료되었습니다.")
    print(f"\n가상 환경을 활성화하려면 다음 명령어를 실행하세요:")
    if platform.system() == 'Windows':
        print(f"  {activate_cmd}")
    else:
        print(f"  {activate_cmd}")
        
    print(f"\n의존성을 설치하려면 다음 명령어를 실행하세요:")
    print(f"  {pip_cmd} install -r {req_path}")
    
    return True

def install_dependencies(model_name, auto_install=False):
    """
    지정된 모델에 대한 의존성을 설치합니다.
    
    Args:
        model_name (str): 모델 이름 (디렉토리 이름)
        auto_install (bool): 자동 설치 여부
    """
    # 개발자 디렉토리 내의 모델 경로 설정
    model_path = os.path.join(DEVELOPER_NAME, model_name)
    venv_path = os.path.join(model_path, 'venv')
    req_path = os.path.join(model_path, 'requirements.txt')
    
    if not os.path.exists(venv_path):
        print(f"오류: 가상 환경이 존재하지 않습니다: {venv_path}")
        return False
    
    if not os.path.exists(req_path):
        print(f"오류: requirements.txt 파일이 존재하지 않습니다: {req_path}")
        return False
    
    if auto_install:
        print(f"{model_name} 모델의 의존성을 설치합니다...")
        
        if platform.system() == 'Windows':
            pip_path = os.path.join(venv_path, 'Scripts', 'pip')
        else:
            pip_path = os.path.join(venv_path, 'bin', 'pip')
        
        try:
            subprocess.run([pip_path, 'install', '-r', req_path], check=True)
            print("의존성 설치가 완료되었습니다.")
            return True
        except subprocess.CalledProcessError:
            print("의존성 설치 중 오류가 발생했습니다.")
            return False
    else:
        if platform.system() == 'Windows':
            activate_cmd = f"{venv_path}\\Scripts\\activate && pip install -r {req_path}"
        else:
            activate_cmd = f"source {venv_path}/bin/activate && pip install -r {req_path}"
        
        print(f"다음 명령어를 사용하여 의존성을 설치하세요:")
        print(f"  {activate_cmd}")
    
    return True

def list_models():
    """
    사용 가능한 모델 목록을 표시합니다.
    """
    # 개발자 디렉토리 확인
    developer_path = DEVELOPER_NAME
    if not os.path.exists(developer_path):
        print(f"오류: {developer_path} 디렉토리가 존재하지 않습니다.")
        print(f"힌트: 'mkdir {developer_path}' 명령어로 개발자 디렉토리를 생성하세요.")
        return False
    
    # 디버깅 정보
    print(f"{developer_path} 디렉토리 내용:")
    
    # 디렉토리가 비어있는지 확인
    if not os.path.exists(developer_path) or not os.listdir(developer_path):
        print("  - 디렉토리가 비어있거나 존재하지 않습니다.")
        return False
    
    for item in os.listdir(developer_path):
        item_path = os.path.join(developer_path, item)
        is_dir = os.path.isdir(item_path)
        has_req = os.path.exists(os.path.join(item_path, 'requirements.txt'))
        print(f"  - {item} (디렉토리: {is_dir}, requirements.txt: {has_req})")
    
    # 첫 번째 레벨 디렉토리 목록 가져오기
    models = []
    for item in os.listdir(developer_path):
        item_path = os.path.join(developer_path, item)
        if os.path.isdir(item_path):
            # requirements.txt 파일이 있는지 확인
            req_path = os.path.join(item_path, 'requirements.txt')
            if os.path.exists(req_path):
                models.append(item)
            else:
                print(f"  - {item} 디렉토리에 requirements.txt 파일이 없습니다.")
    
    if not models:
        print("\n사용 가능한 모델이 없습니다.")
        return False
    
    print("\n사용 가능한 모델 목록:")
    for i, model in enumerate(models, 1):
        req_path = os.path.join(developer_path, model, 'requirements.txt')
        venv_path = os.path.join(developer_path, model, 'venv')
        
        status = []
        if os.path.exists(req_path):
            status.append("요구사항 있음")
        else:
            status.append("요구사항 없음")
        
        if os.path.exists(venv_path):
            status.append("가상 환경 설치됨")
        else:
            status.append("가상 환경 없음")
        
        print(f"  {i}. {model} ({', '.join(status)})")
    
    return True

def check_active_venv():
    """
    현재 활성화된 가상 환경 정보를 확인합니다.
    """
    # 가상 환경 활성화 여부 확인
    venv_path = os.environ.get('VIRTUAL_ENV')
    
    if venv_path:
        venv_name = os.path.basename(os.path.dirname(venv_path))
        python_version = platform.python_version()
        pip_path = os.path.join(venv_path, 'Scripts', 'pip') if platform.system() == 'Windows' else os.path.join(venv_path, 'bin', 'pip')
        
        print("\n현재 활성화된 가상 환경 정보:")
        print(f"  - 가상 환경 경로: {venv_path}")
        print(f"  - 가상 환경 이름: {venv_name}")
        print(f"  - Python 버전: {python_version}")
        
        # 설치된 패키지 정보 확인
        print("\n설치된 주요 패키지:")
        try:
            # pip list 명령어 실행
            result = subprocess.run([pip_path, "list"], capture_output=True, text=True)
            packages = result.stdout.strip().split('\n')[2:]  # 헤더 제외
            
            for package in packages[:10]:  # 상위 10개만 표시
                print(f"  - {package}")
            
            if len(packages) > 10:
                print(f"  - ... 외 {len(packages) - 10}개")
        except Exception as e:
            print(f"  패키지 정보를 가져오는 중 오류 발생: {e}")
        
        # 모델 정보 확인
        developer_path = DEVELOPER_NAME
        if os.path.exists(developer_path):
            for model in os.listdir(developer_path):
                model_venv = os.path.join(developer_path, model, 'venv')
                if os.path.exists(model_venv) and os.path.samefile(venv_path, model_venv):
                    print(f"\n이 가상 환경은 '{DEVELOPER_NAME}/{model}' 모델을 위한 환경입니다.")
                    break
    else:
        print("\n현재 활성화된 가상 환경이 없습니다.")
        print("다음 명령어로 가상 환경을 활성화할 수 있습니다:")
        
        if platform.system() == 'Windows':
            print(f"  {DEVELOPER_NAME}\\[model]\\venv\\Scripts\\activate")
        else:
            print(f"  source {DEVELOPER_NAME}/[model]/venv/bin/activate")
    
    return True

def show_config():
    """
    현재 설정을 보여줍니다.
    """
    # 개발자 이름 다시 가져오기
    developer_name = get_developer_name()
    
    print("\n현재 설정:")
    print(f"  - 개발자 이름: {developer_name}")
    print("\n설정을 변경하려면 다음 명령어를 사용하세요:")
    print("  python setup_env.py set_developer_name [이름]")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델별 가상 환경 설정 및 관리')
    
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # setup 명령어
    setup_parser = subparsers.add_parser('setup', help='가상 환경 설정')
    setup_parser.add_argument('model', help='설정할 모델 이름 (예: random-forest)')
    
    # install 명령어
    install_parser = subparsers.add_parser('install', help='의존성 설치')
    install_parser.add_argument('model', help='의존성을 설치할 모델 이름 (예: random-forest)')
    install_parser.add_argument('--auto', action='store_true', help='자동 설치 수행')
    
    # list 명령어
    list_parser = subparsers.add_parser('list', help='모델 목록 표시')
    
    # info 명령어
    info_parser = subparsers.add_parser('info', help='현재 활성화된 가상 환경 정보 표시')
    
    # config 명령어
    config_parser = subparsers.add_parser('config', help='현재 설정 정보 표시')
    
    # 개발자 이름 설정 명령어
    set_name_parser = subparsers.add_parser('set_developer_name', help='개발자 이름 설정')
    set_name_parser.add_argument('name', help='설정할 개발자 이름')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment(args.model)
    elif args.command == 'install':
        install_dependencies(args.model, args.auto)
    elif args.command == 'list':
        list_models()
    elif args.command == 'info':
        check_active_venv()
    elif args.command == 'config':
        show_config()
    elif args.command == 'set_developer_name':
        set_developer_name(args.name)
    else:
        parser.print_help() 