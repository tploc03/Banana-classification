import cv2
import os

def extract_frames(video_path, output_dir, skip=3, label=""):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    img_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_paths = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            img_path = os.path.join(
                output_dir, f"{video_name}_{img_count+1:02}.jpg"
            )
            cv2.imwrite(img_path, frame)
            image_paths.append(img_path)
            img_count += 1
        count += 1
    cap.release()
    print(f"Extracted {img_count} images from {os.path.basename(video_path)}")
    return image_paths