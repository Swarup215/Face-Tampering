import cv2
import os

def extract_frames(video_path, output_dir, every=10):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0

    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every == 0:
            cv2.imwrite(
                os.path.join(output_dir, f"{saved_count}.jpg"),
                frame
            )
            saved_count += 1

        # progress display
        if frame_count % 50 == 0:
            percent = (frame_count / total_frames) * 100
            print(f"  Progress: {percent:.1f}% ({frame_count}/{total_frames})", end="\r")

        frame_count += 1

    cap.release()
    print(f"\nSaved {saved_count} frames to {output_dir}")