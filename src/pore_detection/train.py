import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict
import logging
from roboflow import Roboflow
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoboflowPoreDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize the pore dataset from Roboflow data.
        
        Args:
            data_dir: Directory containing the Roboflow dataset
            transform: Optional transforms to apply to the images
        """
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load image paths and annotations
        self.image_paths = []
        self.mask_paths = []
        
        # Load train/val/test data
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            # Load images and masks
            images_dir = os.path.join(split_dir, 'images')
            masks_dir = os.path.join(split_dir, 'masks')  # Roboflow stores masks in a separate directory
            
            for img_file in os.listdir(images_dir):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(images_dir, img_file)
                mask_path = os.path.join(masks_dir, img_file.rsplit('.', 1)[0] + '.png')
                
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        
        logger.info(f"Loaded {len(self.image_paths)} images with masks")
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Resize both image and mask
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        mask = mask.resize((224, 224), Image.Resampling.NEAREST)
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
            
        # Convert mask to tensor and normalize to [0, 1]
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = mask.unsqueeze(0)  # Add channel dimension
        
        return image, mask

class PoreDetectionModel(nn.Module):
    def __init__(self):
        super(PoreDetectionModel, self).__init__()
        
        # Encoder
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
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
        x2 = self.enc2(nn.MaxPool2d(2)(x1))
        x3 = self.enc3(nn.MaxPool2d(2)(x2))
        x4 = self.enc4(nn.MaxPool2d(2)(x3))
        
        # Decoder with skip connections
        x = self.up4(x4)
        x = self.dec4(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        x = self.dec1(x)
        
        return torch.sigmoid(self.final(x))

def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int) -> str:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version
        
    Returns:
        Path to downloaded dataset
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("pytorch")
    
    return dataset.location

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """
    Train the pore detection model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # Log progress
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Training Loss: {train_loss/len(train_loader):.4f}')
        logger.info(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/pore_detection_model.pth')
            
    return model

def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get Roboflow credentials
    api_key = os.getenv('ROBOFLOW_API_KEY')
    workspace = os.getenv('ROBOFLOW_WORKSPACE')
    project = os.getenv('ROBOFLOW_PROJECT')
    version = int(os.getenv('ROBOFLOW_VERSION', '1'))
    
    if not all([api_key, workspace, project]):
        raise ValueError("Missing Roboflow credentials in .env file")
    
    # Download dataset
    logger.info("Downloading dataset from Roboflow...")
    data_dir = download_roboflow_dataset(api_key, workspace, project, version)
    
    # Create datasets
    train_dataset = RoboflowPoreDataset(os.path.join(data_dir, 'train'))
    val_dataset = RoboflowPoreDataset(os.path.join(data_dir, 'valid'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = PoreDetectionModel()
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader)
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main() 