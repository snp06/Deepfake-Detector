import cv2
import os

# ===== SETTINGS =====
REAL_VIDEOS = "dataset/real_videos"
FAKE_VIDEOS = "dataset/fake_videos"
OUTPUT_DIR = "dataset/faces"
IMG_SIZE = 299
FRAME_SKIP = 5   # process every 5th frame (speed vs accuracy)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def process_videos(video_dir, label):
    output_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_path, exist_ok=True)

    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)

        if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        print(f"🎥 Processing: {video_file}")

        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed
            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

                filename = f"{video_file}_f{frame_id}_{saved_count}.jpg"
                save_path = os.path.join(output_path, filename)

                cv2.imwrite(save_path, face)
                saved_count += 1

            frame_id += 1

        cap.release()
        print(f"Saved {saved_count} faces from {video_file}\n")


def main():
    print(" Starting dataset processing...\n")
    
    process_videos(FAKE_VIDEOS, "fake")
    process_videos(REAL_VIDEOS, "real")
    
    print(" DONE! Dataset ready at: dataset/faces")


if __name__ == "__main__":
    main()
