import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
from PIL import Image

# This file is now a utility module for video processing tasks.
# The core application logic has been moved to logic.py.

# Check if a GPU is available and set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device, post_process=False)

def extract_frames(video_path, sample_rate=15):
    """
    Generator function to extract frames from a video file.
    :param video_path: Path to the video file.
    :param sample_rate: The rate at which to sample frames (e.g., 15 means every 15th frame).
    """
    vid_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        frame_count += 1
    vid_capture.release()

def detect_and_crop_faces(frame_img: Image.Image):
    """
    Detects and crops faces from a single frame image.
    :param frame_img: A PIL Image in RGB format.
    :return: A list of dictionaries, each containing a cropped face image and its bounding box.
    """
    boxes, _ = mtcnn.detect(frame_img)

    faces_info = []
    if boxes is not None:
        for box in boxes:
            margin = 20
            box_coords = [
                max(0, box[0] - margin),
                max(0, box[1] - margin),
                min(frame_img.width, box[2] + margin),
                min(frame_img.height, box[3] + margin),
            ]
            face = frame_img.crop(box_coords)
            faces_info.append({"face": face, "box": [int(c) for c in box_coords]})
    return faces_info
