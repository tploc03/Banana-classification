import os
import random
from extract_frames import extract_frames

VIDEO_DIR = "video"
TRAIN_DIRS = ["raw"]
TEST_DIRS = ["raw_to_test"]
SKIP_FRAMES = 3
TEST_RATIO = 0.2  # 20% for test

def move_images(image_paths, dest_dirs, label, is_test=False):
    for img_path in image_paths:
        #split for each folder
        if len(dest_dirs) == 2:
            # 52% for dest_dirs[0], 48% for dest_dirs[1]
            dest_dir = dest_dirs[0] if random.random() < 0.52 else dest_dirs[1]
        else:
            dest_dir = random.choice(dest_dirs)
        dest_dir = os.path.join(dest_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        os.rename(img_path, dest_path)

def process_all_videos():
    for label in ["ripe", "unripe", "rotten"]:
        video_label_dir = os.path.join(VIDEO_DIR, label)
        temp_dir = os.path.join("temp_extract", label)
        os.makedirs(temp_dir, exist_ok=True)

        # extract frames from videos
        all_image_paths = []
        for filename in os.listdir(video_label_dir):
            if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                video_path = os.path.join(video_label_dir, filename)
                image_paths = extract_frames(video_path, temp_dir, skip=SKIP_FRAMES, label=label)
                all_image_paths.extend(image_paths)

        # split image for train/test
        random.shuffle(all_image_paths)
        n_total = len(all_image_paths)
        n_test = int(n_total * TEST_RATIO)
        test_images = all_image_paths[:n_test]
        train_images = all_image_paths[n_test:]

        # move image to folder train/test
        move_images(train_images, TRAIN_DIRS, label, is_test=False)
        move_images(test_images, TEST_DIRS, label, is_test=True)

        #remove temp images
        for img in all_image_paths:
            if os.path.exists(img):
                os.remove(img)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == "__main__":
    process_all_videos()