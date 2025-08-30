import pytest
import numpy as np
from unittest.mock import patch
from PIL import Image
import os
from backend.logic import aggregate_results, detect_and_crop_faces, process_video
from backend.model import DeepfakeDetector

# --- Test Cases for process_video ---

@patch('backend.logic.extract_frames')
@patch('backend.logic.detect_and_crop_faces')
def test_process_video_with_faces(mock_detect_faces, mock_extract_frames):
    """
    Test the main video processing pipeline with mocking.
    """
    # Mock the frame extractor to yield a few dummy frames
    dummy_frame = np.uint8(np.random.rand(100, 100, 3) * 255)
    mock_extract_frames.return_value = [dummy_frame, dummy_frame, dummy_frame]

    # Mock face detection to return a face for the first two frames
    dummy_face_img = Image.new('RGB', (50, 50))
    mock_detect_faces.side_effect = [
        [{"face": dummy_face_img, "box": [0,0,50,50]}],
        [{"face": dummy_face_img, "box": [0,0,50,50]}],
        []  # No face in the third frame
    ]

    # Mock the detector model
    mock_detector = patch('backend.logic.DeepfakeDetector').start()
    mock_detector_instance = mock_detector.return_value
    mock_detector_instance.predict_batch.return_value = [0.1, 0.9]

    # Run the video processing
    scores = process_video("dummy_path.mp4", mock_detector_instance)

    # Assertions
    assert mock_extract_frames.call_count == 1
    assert mock_detect_faces.call_count == 3
    mock_detector_instance.predict_batch.assert_called_once()
    assert scores == [0.1, 0.9]

    patch.stopall()

@patch('backend.logic.extract_frames')
@patch('backend.logic.detect_and_crop_faces')
def test_process_video_no_faces(mock_detect_faces, mock_extract_frames):
    """
    Test the video processing pipeline when no faces are detected.
    """
    dummy_frame = np.uint8(np.random.rand(100, 100, 3) * 255)
    mock_extract_frames.return_value = [dummy_frame]
    mock_detect_faces.return_value = [] # No faces found

    mock_detector = patch('backend.logic.DeepfakeDetector').start()
    mock_detector_instance = mock_detector.return_value

    scores = process_video("dummy_path.mp4", mock_detector_instance)

    mock_detector_instance.predict_batch.assert_not_called()
    assert scores == []

    patch.stopall()


# --- Test Cases for detect_and_crop_faces ---

@patch('backend.logic.mtcnn')
def test_detect_and_crop_faces_with_face(mock_mtcnn):
    """
    Test that faces are detected and cropped correctly from an image.
    Mocks the MTCNN model to avoid heavy computation.
    """
    # Configure the mock to return a fake bounding box
    mock_mtcnn.detect.return_value = ([np.array([10, 20, 50, 60])], None)

    # Create a dummy image. The content doesn't matter as MTCNN is mocked.
    dummy_image = Image.new('RGB', (100, 100), color = 'red')

    faces = detect_and_crop_faces(dummy_image)

    # Check that detect was called
    mock_mtcnn.detect.assert_called_once_with(dummy_image)

    # Check that one face was returned
    assert len(faces) == 1
    # Check that the face has the correct structure
    assert "face" in faces[0]
    assert "box" in faces[0]
    # Check that the returned image is a PIL Image
    assert isinstance(faces[0]["face"], Image.Image)

@patch('backend.logic.mtcnn')
def test_detect_and_crop_faces_no_face(mock_mtcnn):
    """
    Test the case where no faces are detected in the image.
    """
    # Configure the mock to return no bounding boxes
    mock_mtcnn.detect.return_value = (None, None)

    dummy_image = Image.new('RGB', (100, 100), color = 'red')
    faces = detect_and_crop_faces(dummy_image)

    # Check that detect was called
    mock_mtcnn.detect.assert_called_once_with(dummy_image)
    # Check that no faces were returned
    assert len(faces) == 0


# --- Test Cases for aggregate_results ---

def test_aggregate_results_no_scores():
    """
    Test aggregation when no scores are provided (e.g., no faces detected).
    """
    result = aggregate_results([])
    assert result["verdict"] == "NO_FACES_DETECTED"
    assert result["overall_score"] == 0
    assert result["confidence"] == 0

def test_aggregate_results_real_verdict():
    """
    Test aggregation for a video that is clearly real.
    """
    scores = [0.1, 0.15, 0.2, 0.12]
    result = aggregate_results(scores)
    assert result["verdict"] == "REAL"
    assert result["overall_score"] == pytest.approx(np.mean(scores))
    assert len(result["suspicious_frames"]) == 0

def test_aggregate_results_fake_verdict():
    """
    Test aggregation for a video that is clearly fake.
    """
    scores = [0.8, 0.85, 0.9, 0.92]
    result = aggregate_results(scores)
    assert result["verdict"] == "FAKE"
    assert result["overall_score"] == pytest.approx(np.mean(scores))
    assert result["confidence"] == pytest.approx(np.max(scores))
    assert len(result["suspicious_frames"]) == 4

def test_aggregate_results_suspicious_verdict():
    """
    Test aggregation for a video with mixed or borderline scores.
    """
    scores = [0.4, 0.5, 0.6, 0.55]
    result = aggregate_results(scores)
    assert result["verdict"] == "SUSPICIOUS"
    assert result["overall_score"] == pytest.approx(np.mean(scores))
    # One frame has a score > 0.6
    assert len(result["suspicious_frames"]) == 0

def test_aggregate_results_with_suspicious_frames():
    """
    Test that suspicious frames are correctly identified.
    """
    scores = [0.1, 0.9, 0.2, 0.75, 0.3]
    result = aggregate_results(scores)
    assert result["suspicious_frames"] == [1, 3]
