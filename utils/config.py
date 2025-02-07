import yaml
import argparse
from pathlib import Path

def load_config(config_path=None):
    """설정 파일을 로드하고 커맨드 라인 인자와 병합"""
    parser = argparse.ArgumentParser(description='딥페이크 탐지 실험 설정')
    
    # 기본 인자 설정
    parser.add_argument('--config', type=str, default='analysis_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--input_dir', type=str, help='입력 이미지 디렉토리')
    parser.add_argument('--output_dir', type=str, help='결과 저장 디렉토리')
    parser.add_argument('--checkpoint_path', type=str, help='모델 체크포인트 경로')
    parser.add_argument('--face_detector', type=str, help='dlib 얼굴 감지기 모델 경로')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config_path = config_path or args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 커맨드 라인 인자로 설정 덮어쓰기
    if args.input_dir:
        config['paths']['input_dir'] = args.input_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.checkpoint_path:
        config['paths']['checkpoint_path'] = args.checkpoint_path
    if args.face_detector:
        config['paths']['face_detector'] = args.face_detector
    
    # 필수 경로 확인
    required_paths = ['input_dir', 'checkpoint_path', 'face_detector']
    for path in required_paths:
        if not config['paths'][path]:
            raise ValueError(f"'{path}' 경로가 설정되지 않았습니다.")
    
    # 출력 디렉토리 생성
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    return config 