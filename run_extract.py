import os
from extract_frames import extract_frames

print("=== Extracting REAL videos ===")
for video in os.listdir("real"):
    extract_frames(
        os.path.join("real", video),
        os.path.join("frames", "real", video[:-4])
    )

print("\n=== Extracting FAKE videos ===")
for video in os.listdir("fake"):
    extract_frames(
        os.path.join("fake", video),
        os.path.join("frames", "fake", video[:-4])
    )

print("\nFrame extraction completed.")