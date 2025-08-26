# 🎯 Deepfake Radar

Deepfake Radar is a web application designed to detect deepfake videos using AI. It provides a user-friendly interface to upload a video and receive an analysis of its authenticity, including an overall verdict and a frame-by-frame probability score.

This project was developed as a full-stack application featuring a Python backend with FastAPI and a frontend built with Streamlit. The entire application is containerized using Docker for easy setup and deployment.

## ✨ Features

-   **AI-Powered Detection**: Utilizes a pre-trained XceptionNet model to analyze video frames for signs of manipulation.
-   **Face Detection**: Employs MTCNN for accurate face detection in video frames.
-   **Interactive Frontend**: A Streamlit dashboard allows for easy video uploads and clear visualization of results.
-   **Detailed Analysis**: Provides an overall verdict (REAL, SUSPICIOUS, FAKE), a confidence score, and a frame-by-frame probability chart.
-   **Containerized**: Fully containerized with Docker and Docker Compose for one-command setup.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You must have the following software installed on your machine:
-   [Docker](https://www.docker.com/get-started)
-   [Docker Compose](https://docs.docker.com/compose/install/)

### Installation & Launch

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Launch the application using Docker Compose:**
    From the root directory of the project, run the following command:
    ```sh
    docker-compose up --build
    ```
    This command will build the Docker images for the frontend and backend services and then start the containers. The `--build` flag ensures that the images are rebuilt if there are any changes to the Dockerfiles or the code.

3.  **Access the application:**
    Once the containers are running, you can access the different parts of the application:
    -   **Frontend Application**: Open your web browser and navigate to `http://localhost:8501`
    -   **Backend API Docs**: The FastAPI backend provides automatically generated documentation.
        -   **Swagger UI**: `http://localhost:8000/docs`
        -   **ReDoc**: `http://localhost:8000/redoc`

## 🛠️ Built With

-   **Backend**:
    -   [FastAPI](https://fastapi.tiangolo.com/) - The web framework for building APIs.
    -   [PyTorch](https://pytorch.org/) - The deep learning framework.
    -   [Timm](https://timm.fast.ai/) - For pre-trained computer vision models.
    -   [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - For MTCNN face detection.
-   **Frontend**:
    -   [Streamlit](https://streamlit.io/) - The framework for building the web app.
-   **Deployment**:
    -   [Docker](https://www.docker.com/) - For containerization.
    -   [Docker Compose](https://docs.docker.com/compose/) - For orchestrating multi-container applications.
