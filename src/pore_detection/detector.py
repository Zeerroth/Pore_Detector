import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoreDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> nn.Module:
        """Load the pre-trained pore detection model."""
        # Simple U-Net like architecture for pore detection
        class PoreDetectionModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = self._block(3, 64)
                self.enc2 = self._block(64, 128)
                self.enc3 = self._block(128, 256)
                
                # Decoder
                self.dec3 = self._block(256, 128)
                self.dec2 = self._block(128, 64)
                self.dec1 = self._block(64, 32)
                
                # Final layer
                self.final = nn.Conv2d(32, 1, kernel_size=1)
                
            def _block(self, in_ch: int, out_ch: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Encoder
                x1 = self.enc1(x)
                x2 = self.enc2(F.max_pool2d(x1, 2))
                x3 = self.enc3(F.max_pool2d(x2, 2))
                
                # Decoder
                x = F.interpolate(x3, scale_factor=2)
                x = self.dec3(x + x2)
                x = F.interpolate(x, scale_factor=2)
                x = self.dec2(x + x1)
                x = self.dec1(x)
                
                return torch.sigmoid(self.final(x))

        return PoreDetectionModel()

    def detect_pores(self, image: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect pores in the input image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing pore coordinates and confidence scores
        """
        # Preprocess image
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            pred = self.model(image)
            pred = pred.squeeze().cpu().numpy()
        
        # Threshold prediction
        binary_mask = (pred > 0.5).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        pores = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < 10:  # Filter out tiny contours
                continue
                
            # Get center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pores.append((cx, cy))
        
        return {
            "pore_coordinates": pores,
            "confidence_scores": [float(pred[y, x]) for x, y in pores]
        }

    def generate_mask(self, image: np.ndarray, pore_data: Dict[str, List]) -> np.ndarray:
        """
        Generate a visual mask highlighting detected pores.
        
        Args:
            image: Original image
            pore_data: Dictionary containing pore coordinates and confidence scores
            
        Returns:
            Image with pore highlights
        """
        mask = image.copy()
        
        # Draw circles around detected pores
        for (x, y), score in zip(pore_data["pore_coordinates"], pore_data["confidence_scores"]):
            # Scale coordinates back to original image size
            x = int(x * image.shape[1] / 256)
            y = int(y * image.shape[0] / 256)
            
            # Draw circle with color based on confidence
            color = (0, int(255 * score), 0)  # Green with intensity based on confidence
            cv2.circle(mask, (x, y), 3, color, -1)
        
        # Create semi-transparent overlay
        overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        
        return overlay 