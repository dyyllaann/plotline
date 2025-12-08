"""
CNN model for mold classification.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class MoldCNN(nn.Module):
    """Simple CNN for mold classification."""
    def __init__(self):
        super(MoldCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 classes: mold, not mold
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CNNClassifier:
    """Wrapper for CNN model inference."""
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoldCNN().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def classify_region(self, image, contour, padding=10):
        """
        Classify if a region contains mold.
        
        Args:
            image: BGR image
            contour: Contour to classify
            padding: Extra pixels around contour
            
        Returns:
            is_mold: Boolean
            confidence: Float (0-1)
        """
        # Extract region
        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        region = image[y1:y2, x1:x2]
        
        # Convert to PIL and transform
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        region_pil = Image.fromarray(region_rgb)
        region_tensor = self.transform(region_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(region_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0][1].item()  # Probability of mold class
            is_mold = confidence > 0.5
        
        return is_mold, confidence