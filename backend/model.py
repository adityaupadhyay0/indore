import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

class DeepfakeDetector(nn.Module):
    def __init__(self, device):
        super(DeepfakeDetector, self).__init__()
        self.device = device

        # 1. Load the pretrained Xception model from timm
        print("Loading pretrained Xception model...")
        self.model = timm.create_model('xception', pretrained=True)

        # 2. Replace the final classifier layer for binary classification (real/fake)
        # This is a standard transfer learning technique.
        num_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(num_features, 2) # 2 classes: 0 for real, 1 for fake

        # In a full project, you would now load your fine-tuned weights, e.g.:
        # self.model.load_state_dict(torch.load('path/to/your_finetuned_weights.pth'))
        # For this MVP, we will use the ImageNet backbone with a randomly initialized final layer.
        # This still provides a much stronger feature representation than a random model.

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded and set to evaluation mode.")

        # 3. Define the image transformations required by the Xception model
        self.transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            # Normalization values for ImageNet-pretrained models
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, face_image: Image.Image) -> float:
        """
        Predicts if a face image is a deepfake.
        Returns the probability of the image being 'fake'.
        """
        if face_image is None:
            return 0.0

        # Apply transformations and add a batch dimension
        image_tensor = self.transforms(face_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # We assume class 1 corresponds to 'fake'
            fake_probability = probabilities[0][1].item()

        return fake_probability

    def predict_batch(self, face_images: list[Image.Image]) -> list[float]:
        """
        Predicts if a batch of face images are deepfakes.
        Returns a list of probabilities of the images being 'fake'.
        """
        if not face_images:
            return []

        # Transform all images and stack them into a single batch tensor
        batch_tensor = torch.stack(
            [self.transforms(image) for image in face_images]
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # Get the probability for the 'fake' class (class 1) for each image
            fake_probabilities = probabilities[:, 1].tolist()

        return fake_probabilities
