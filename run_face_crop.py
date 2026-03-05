import os
from face_crop import crop_faces

print("=== Cropping REAL faces ===")
for video_folder in os.listdir("frames/real"):
    crop_faces(
        os.path.join("frames", "real", video_folder),
        os.path.join("faces", "real", video_folder)
    )

print("\n=== Cropping FAKE faces ===")
for video_folder in os.listdir("frames/fake"):
    crop_faces(
        os.path.join("frames", "fake", video_folder),
        os.path.join("faces", "fake", video_folder)
    )

print("\nFace cropping completed.")