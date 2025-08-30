import pytest
import httpx
from unittest.mock import patch
from fastapi.testclient import TestClient

# It's important that the relative imports are correct
from backend.main import app

client = TestClient(app)

def test_health_check():
    """
    Tests the health check endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# The patch needs to target the 'process_video' function where it's used, which is in 'main'
@patch('backend.main.process_video')
@patch('backend.main.aggregate_results')
def test_analyze_video_success(mock_aggregate_results, mock_process_video):
    """
    Tests a successful video analysis request.
    Mocks the heavy processing functions.
    """
    # Configure mocks to return predictable results
    mock_process_video.return_value = [0.1, 0.2, 0.9] # Dummy frame scores
    mock_aggregate_results.return_value = {
        "overall_score": 0.4,
        "verdict": "SUSPICIOUS",
        "confidence": 0.9,
        "suspicious_frames": [2]
    }

    video_path = "backend/test_video.mp4"
    with open(video_path, "rb") as video_file:
        files = {"file": ("test_video.mp4", video_file, "video/mp4")}
        response = client.post("/analyze-video", files=files)

    assert response.status_code == 200
    result = response.json()
    assert result["filename"] == "test_video.mp4"
    assert result["verdict"] == "SUSPICIOUS"
    assert "frame_scores" in result
    assert len(result["frame_scores"]) == 3

    # Ensure the mocked functions were called
    mock_process_video.assert_called_once()
    mock_aggregate_results.assert_called_once_with([0.1, 0.2, 0.9])

def test_analyze_video_no_file():
    """
    Test hitting the endpoint without a file.
    FastAPI should return a 422 Unprocessable Entity error.
    """
    response = client.post("/analyze-video")
    assert response.status_code == 422 # Unprocessable Entity
