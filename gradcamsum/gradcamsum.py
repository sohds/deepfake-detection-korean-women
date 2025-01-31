import os
import numpy as np
from PIL import Image, ImageEnhance

def combine_and_enhance_images(base_dir, output_dir, subfolder_prefix, num_images=165, scaling_factor=2):
    """
    주어진 디렉토리에서 이미지를 불러와 평균 합성한 후 색상을 조정하여 저장하는 함수.

    :param base_dir: 이미지들이 저장된 기본 디렉토리 경로
    :param output_dir: 결과 이미지를 저장할 디렉토리 경로
    :param subfolder_prefix: 각 서브폴더 이름의 접두사 
    :param num_images: 처리할 이미지 수 (기본값 165)
    :param scaling_factor: 색상을 더 진하게 하기 위한 스케일링 팩터 (기본값 2)
    """
    # 결과를 저장할 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 0부터 num_images까지 반복
    for i in range(num_images):
        dir_path = os.path.join(base_dir, f"{subfolder_prefix}_{i}")

        # 경로에 있는 이미지 파일 리스트 가져오기
        image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 이미지 불러오기 및 배열로 변환
        images = [Image.open(os.path.join(dir_path, f)).convert('RGBA') for f in image_files]

        # 첫 번째 이미지의 크기 가져오기 (모든 이미지가 같은 크기라고 가정)
        if not images:  # 이미지가 없는 경우 다음 폴더로 넘어감
            continue
        width, height = images[0].size

        # 모든 이미지를 numpy 배열로 변환하여 합산
        combined_array = np.zeros((height, width, 4), dtype=np.float32)

        for img in images:
            img_array = np.array(img, dtype=np.float32) / 255.0  # 0-255 값을 0-1 사이로 정규화
            combined_array += img_array

        # 평균 값으로 합산된 이미지 생성
        combined_array /= len(images)

        # 0-1 사이 값을 다시 0-255로 변환
        combined_array = (combined_array * 255).astype(np.uint8)

        # numpy 배열을 다시 이미지로 변환
        combined_image = Image.fromarray(combined_array, mode='RGBA')

        # 색상을 더 진하게 하기 위해 ImageEnhance 사용
        enhancer = ImageEnhance.Color(combined_image)
        combined_image_enhanced = enhancer.enhance(scaling_factor)

        # 결과 이미지 저장 경로 설정
        combined_image_filename = f"combined_image_{i}.png"
        combined_image_enhanced.save(os.path.join(output_dir, combined_image_filename))

    print("모든 결과 이미지가 저장되었습니다.")
