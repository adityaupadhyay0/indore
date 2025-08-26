from PIL import Image

# Use a relative import now that we are running as a module
from .logic import get_deepfake_detector, process_image

def run_image_test():
    """
    Tests the deepfake detector on a single synthetic face image.
    """
    print("--- Starting Single Image Test ---")

    image_path = "backend/test_face.jpg"
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Test image not found at {image_path}")
        print("Please ensure the test image exists.")
        return

    print(f"Loading image from: {image_path}")

    # Get the detector instance
    detector = get_deepfake_detector()

    # Analyze the image using the dedicated function
    print("Analyzing image...")
    result = process_image(image, detector)

    print("\n--- Test complete. Image Analysis Result: ---")
    if result is None:
        print("Verdict: NO_FACE_DETECTED")
    else:
        score = result["score"]
        # A simple threshold can be used for a binary verdict
        verdict = "FAKE" if score > 0.5 else "REAL"
        print(f"  - Detected Face Bounding Box: {result['box']}")
        print(f"  - Deepfake Score (probability of fake): {score:.4f}")
        print(f"  - Verdict: {verdict}")

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_image_test()
