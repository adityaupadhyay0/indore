import streamlit as st
import requests
import time
import pandas as pd
import os

# --- Configuration ---
# Get the backend URL from environment variables, with a fallback for local development
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/analyze-video"

# --- UI Elements ---
st.set_page_config(page_title="Deepfake Radar", page_icon="🎯", layout="wide")

st.title("🎯 Deepfake Radar")
st.subheader("AI-Powered Video Authenticity Checker")
st.markdown("""
Upload a video file (MP4, AVI, MOV) to analyze it for deepfake manipulations.
The AI model will process the video, detect faces, and provide a frame-by-frame analysis.
""")

uploaded_file = st.file_uploader("Upload Your Video Here", type=['mp4', 'avi', 'mov'])

def analyze_video_api(file_bytes, filename, content_type):
    """
    Sends the video file to the backend API for analysis.
    Handles connection errors and API errors gracefully.
    """
    files = {"file": (filename, file_bytes, content_type)}
    try:
        response = requests.post(BACKEND_URL, files=files, timeout=300)
        # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend API. Please ensure the backend server is running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: An unexpected error occurred. {e}")
        return None

# Set a maximum file size in bytes (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File is too large. Please upload a video under {MAX_FILE_SIZE // (1024 * 1024)}MB.")
    else:
        st.video(uploaded_file)

        file_bytes = uploaded_file.getvalue()
        file_details = {"filename": uploaded_file.name, "content_type": uploaded_file.type}

        if st.button("Analyze Video"):
            spinner_text = "🤖 AI is analyzing the video... This may take a moment. Please wait."
            with st.spinner(spinner_text):
                result = analyze_video_api(file_bytes, file_details["filename"], file_details["content_type"])

                if result:
                    st.subheader("📊 Analysis Results")

                    verdict = result.get("verdict", "UNKNOWN")
                overall_score = result.get("overall_score", 0)
                confidence = result.get("confidence", 0) * 100
                suspicious_frames = result.get("suspicious_frames", [])

                # --- Display Verdict ---
                col1, col2 = st.columns(2)
                with col1:
                    if verdict == "REAL":
                        st.success(f"## ✅ Verdict: REAL")
                    elif verdict == "SUSPICIOUS":
                        st.warning(f"## 🟡 Verdict: SUSPICIOUS")
                    elif verdict == "FAKE":
                        st.error(f"## 🔴 Verdict: FAKE")
                    elif verdict == "NO_FACES_DETECTED":
                        st.info("## ℹ️ No faces were detected.")
                    else:
                        st.info(f"## {verdict}")

                with col2:
                    st.metric(label="Overall Fake Probability", value=f"{overall_score*100:.2f}%")
                    st.metric(label="Confidence in Verdict", value=f"{confidence:.2f}%")

                # --- Display Suspicious Frame Info ---
                if suspicious_frames:
                    st.warning(f"**Suspicious Activity Detected!**")
                    st.markdown(f"The model identified **{len(suspicious_frames)}** frames with a high probability of being manipulated. These are the frames that most contributed to the overall score.")
                    st.info(f"Suspicious frame numbers (approximate): {', '.join(map(str, suspicious_frames))}")


                # --- Display Detailed Analysis ---
                with st.expander("Show Detailed Frame-by-Frame Analysis"):
                    st.markdown("This chart shows the 'fake' probability for each frame sequence analyzed in the video.")
                    frame_scores = result.get("frame_scores")
                    if frame_scores:
                        chart_data = pd.DataFrame({
                            "Frame Sequence": range(len(frame_scores)),
                            "Fake Probability": frame_scores
                        })
                        st.line_chart(chart_data, x="Frame Sequence", y="Fake Probability")
                    else:
                        st.write("No frame-level scores were generated.")

                    st.markdown("---")
                    st.markdown("**Raw API Response:**")
                    st.json(result)
            else:
                st.error("Failed to get a response from the analysis server.")
