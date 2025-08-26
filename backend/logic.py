import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

from .model import DeepfakeDetector

# --- Model and Device Setup ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
detector_instance = None

def get_deepfake_detector():
    """
    Returns a single, shared instance of the DeepfakeDetector.
    Initializes the model on the first call.
    """
    global detector_instance
    if detector_instance is None:
        print(f"Initializing DeepfakeDetector on device: {device}")
        detector_instance = DeepfakeDetector(device=device)
    return detector_instance

# --- Frame Extraction and Face Detection ---
def extract_frames(video_path, sample_rate=15):
    """
    Generator function to extract frames from a video file.
    """
    vid_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        frame_count += 1
    vid_capture.release()

def detect_and_crop_faces(frame_img):
    """
    Detects and crops faces from a single frame image.
    :param frame_img: A PIL Image.
    :return: A list of dictionaries, each containing a cropped face image and its bounding box.
    """
    boxes, _ = mtcnn.detect(frame_img)

    faces = []
    if boxes is not None:
        for box in boxes:
            margin = 20
            box_coords = [
                max(0, box[0]-margin),
                max(0, box[1]-margin),
                min(frame_img.width, box[2]+margin),
                min(frame_img.height, box[3]+margin),
            ]
            face = frame_img.crop(box_coords)
            faces.append({"face": face, "box": [int(c) for c in box]})
    return faces


# --- Core Processing Logic ---
def process_image(image: Image.Image, detector: DeepfakeDetector):
    """
    Processes a single image to detect deepfakes.
    Detects the most prominent face and returns its deepfake score.
    :return: A dictionary with score, box, or None if no face is detected.
    """
    faces_info = detect_and_crop_faces(image)
    if not faces_info:
        return None

    # For simplicity, we analyze the first detected face.
    # A more robust solution might analyze all faces.
    face_info = faces_info[0]
    score = detector.predict(face_info["face"])

    return {"score": score, "box": face_info["box"]}

def process_video(video_path: str, detector: DeepfakeDetector, batch_size: int = 32):
    """
    Processes a video using batching to improve performance.
    It extracts faces, groups them into batches, and sends them for prediction.
    :param video_path: Path to the video file.
    :param detector: An instance of DeepfakeDetector.
    :param batch_size: The number of faces to process in a single batch.
    :return: A list of scores for frames that contained faces.
    """
    faces_to_predict = []
    print("Step 1: Extracting frames and detecting faces...")
    for frame in extract_frames(video_path):
        frame_img = Image.fromarray(frame)
        # For simplicity, we only analyze the first detected face per frame.
        faces_info = detect_and_crop_faces(frame_img)
        if faces_info:
            faces_to_predict.append(faces_info[0]["face"])

    if not faces_to_predict:
        print("No faces were detected in the video.")
        return []

    print(f"Step 2: Running batch prediction on {len(faces_to_predict)} detected faces...")
    all_scores = []
    # Process faces in batches to manage memory
    for i in range(0, len(faces_to_predict), batch_size):
        batch = faces_to_predict[i:i + batch_size]
        scores = detector.predict_batch(batch)
        all_scores.extend(scores)
        print(f"  - Processed batch {i//batch_size + 1}/{-(-len(faces_to_predict)//batch_size)}")

    print("Batch prediction complete.")
    return all_scores

# --- Result Aggregation ---
def aggregate_results(scores):
    """
    Aggregates frame-level scores into an overall verdict.
    """
    if not scores:
        return {
            "overall_score": 0,
            "verdict": "NO_FACES_DETECTED",
            "confidence": 0,
            "suspicious_frames": []
        }

    overall_score = np.mean(scores)
    confidence = np.max(scores) if overall_score > 0.5 else 1 - np.min(scores)

    if overall_score > 0.7:
        verdict = "FAKE"
    elif overall_score > 0.3:
        verdict = "SUSPICIOUS"
    else:
        verdict = "REAL"

    suspicious_frames_indices = [i for i, score in enumerate(scores) if score > 0.6]

    return {
        "overall_score": float(overall_score),
        "verdict": verdict,
        "confidence": float(confidence),
        "suspicious_frames": suspicious_frames_indices
    }
