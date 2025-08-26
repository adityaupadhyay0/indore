import cv2
import numpy as np

def generate_blank_video(filename="test_video.mp4", width=128, height=128, duration_secs=2, fps=30):
    """Generates a short, blank video file using OpenCV."""
    # Using 'mp4v' codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

    num_frames = duration_secs * fps
    for _ in range(num_frames):
        # Create a blank black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)

    print(f"Successfully generated blank video: {filename}")
    out.release()

if __name__ == "__main__":
    # Generate the video in the backend directory
    generate_blank_video("backend/test_video.mp4")
