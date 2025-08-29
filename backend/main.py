from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import time
import numpy as np

# Import the refactored logic
from logic import get_deepfake_detector, process_video, aggregate_results
from model import DeepfakeDetector

app = FastAPI(
    title="Deepfake Radar API",
    description="""
This API provides access to the Deepfake Radar's analysis engine.
You can upload a video file to have it analyzed for deepfake manipulations.
The API will return a detailed analysis including a verdict, confidence scores,
and frame-by-frame probabilities.
    """,
    version="1.0.0",
)

# Allow all origins for development purposes
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for temporary file uploads
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post(
    "/analyze-video",
    summary="Analyze a video for deepfakes",
    description="""
Upload a video file (MP4, AVI, MOV) to start a deepfake analysis.
The video is processed frame by frame to detect faces and analyze them using
an AI model. The endpoint returns a JSON object with the analysis results.
    """,
)
async def analyze_video(
    file: UploadFile = File(..., description="The video file to be analyzed."),
    detector: DeepfakeDetector = Depends(get_deepfake_detector),
):
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    start_time = time.time()

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use the centralized logic functions
        frame_scores = process_video(temp_file_path, detector)
        analysis_result = aggregate_results(frame_scores)

        processing_time = time.time() - start_time

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            **analysis_result,
            "frame_scores": [float(score) for score in frame_scores],
            "processing_time": round(processing_time, 2)
        }
    finally:
        # Clean up the uploaded file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get(
    "/health",
    summary="Health Check",
    description="A simple endpoint to verify that the API is running and accessible.",
)
async def health_check():
    return {"status": "ok"}
