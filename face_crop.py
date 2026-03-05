import cv2
import os

# Load Viola-Jones face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_faces(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)
    total = len(images)
    saved = 0

    print(f"  Processing folder: {input_dir}")
    print(f"  Total frames: {total}")

    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (256, 256))

        cv2.imwrite(os.path.join(output_dir, img_name), face)
        saved += 1

        # progress update every 20 frames
        if idx % 20 == 0:
            percent = (idx / total) * 100 if total > 0 else 0
            print(f"    Progress: {percent:.1f}% ({idx}/{total})", end="\r")

    print(f"\n  Saved faces: {saved}/{total}")