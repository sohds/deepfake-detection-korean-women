import cv2
import dlib
import numpy as np

def crop_facial_features(image_path, output_size=(64, 64)):
    # Load the detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None

    # Get the first face
    face = faces[0]

    # Get facial landmarks
    landmarks = predictor(gray, face)

    # Define landmark indices for eyes, nose, and mouth
    left_eye = list(range(36, 42))
    right_eye = list(range(42, 48))
    nose = list(range(27, 36))
    mouth = list(range(48, 68))

    def crop_feature(feature_indices):
        points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in feature_indices])
        padding = 10
        x, y = np.min(points, axis=0) - padding
        x_max, y_max = np.max(points, axis=0) + padding
        cropped = img[max(0, y):y_max, max(0, x):x_max]
        return cv2.resize(cropped, output_size)

    # Crop features
    left_eye_crop = crop_feature(left_eye)
    right_eye_crop = crop_feature(right_eye)
    nose_crop = crop_feature(nose)
    mouth_crop = crop_feature(mouth)

    return {
        "left_eye": left_eye_crop,
        "right_eye": right_eye_crop,
        "nose": nose_crop,
        "mouth": mouth_crop
    }

# Example usage
# features = crop_facial_features("path/to/your/image.jpg")
# if features:
#     for feature_name, feature_img in features.items():
#         cv2.imwrite(f"{feature_name}.jpg", feature_img)