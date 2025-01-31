#일본인 크롭
import os
import dlib
import cv2
import argparse

# 커맨드 라인 인자 파싱 함수
def parse_args():
    parser = argparse.ArgumentParser(description='얼굴 영역을 크롭하여 데이터셋 생성')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='입력 비디오가 있는 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='출력 이미지를 저장할 디렉토리 경로')
    parser.add_argument('--max_frames', type=int, default=150,
                        help='비디오당 추출할 최대 프레임 수 (기본값: 150)')
    parser.add_argument('--scale_factor', type=float, default=1.7,
                        help='얼굴 영역 확장 비율 (기본값: 1.7)')
    parser.add_argument('--margin', type=int, default=10,
                        help='얼굴 영역 주변 여백 (기본값: 10)')
    return parser.parse_args()

# 크롭할 video 디렉토리 경로
dir_path = ""

folder_list = []

for f in os.listdir(dir_path):
    folder_list.append(os.path.join(dir_path, f))

print(folder_list)

# dlib 얼굴 감지 결과를 OpenCV 바운딩 박스로 변환
def convert_and_trim_bb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    w = endX - startX
    h = endY - startY
    return (startX, startY, w, h)

# 비디오에서 프레임 추출하고 얼굴 영역 저장
def get_frames(input_path, output_path, max_frames=150):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"에러: {input_path} 파일을 열 수 없습니다.")
        return False
    
    frame_id = 0
    frame_skip = 1
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id > frame_skip - 1:
                frame_count += 1
                boxes = get_face_landmarks(frame)
                if len(boxes) > 0:
                    try:
                        cropped = get_cropped_image(frame, boxes)
                        if cropped is not None:
                            output_file = f'frame_{frame_count:04d}.jpg'
                            output_frame_path = os.path.join(output_path, output_file)
                            cv2.imwrite(output_frame_path, cropped)
                            print(f"저장됨: {output_frame_path}")
                    except Exception as e:
                        print(f"프레임 처리 중 에러 발생: {e}")
                        continue
                
                frame_id = 0
                if frame_count >= max_frames:
                    break
            frame_id += 1
    except Exception as e:
        print(f"비디오 처리 중 에러 발생: {e}")
        return False
    finally:
        cap.release()
    return True

# 이미지에서 얼굴 영역 감지
def get_face_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rects = detector(rgb)
    boxes = [convert_and_trim_bb(image, r) for r in rects]
    return boxes

# 감지된 얼굴 영역으로 이미지 크롭
def get_cropped_image(image, boxes, scale_factor=1.7, margin=10):
    height, width = image.shape[0], image.shape[1]
    for (x, y, w, h) in boxes:
        center_x, center_y = x + w // 2, y + h // 2
        new_w = int(w * scale_factor + margin)
        new_h = int(h * scale_factor + margin)
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        new_x2 = min(new_x + new_w, width)
        new_y2 = min(new_y + new_h, height)
        cropped = image[new_y:new_y2, new_x:new_x2]
        return cropped
    return None

# 메인 실행 함수
def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"에러: 입력 디렉토리가 존재하지 않습니다: {args.input_dir}")
        return
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    video_files = [f for f in os.listdir(args.input_dir) 
                   if os.path.isfile(os.path.join(args.input_dir, f))]
    
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(args.input_dir, video_file)
        output_path = os.path.join(args.output_dir, f"fake_ja_{idx}")
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        print(f"처리 중: {video_path}")
        success = get_frames(video_path, output_path, args.max_frames)
        if success:
            print(f"완료: {output_path}")
        else:
            print(f"실패: {video_path}")

if __name__ == "__main__":
    main()
