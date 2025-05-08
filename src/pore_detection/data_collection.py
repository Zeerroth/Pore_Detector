import os
import cv2
import numpy as np
from typing import List, Tuple
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: str):
        """
        Initialize the data collector.
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.annotations_dir = self.data_dir / 'annotations'
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_image(self, image_path: str, facial_region: str) -> str:
        """
        Collect and save an image for pore detection.
        
        Args:
            image_path: Path to the input image
            facial_region: Region of the face (e.g., 'cheek', 'nose', 'forehead')
            
        Returns:
            Path to the saved image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Generate unique filename
        filename = f"{facial_region}_{len(list(self.images_dir.glob('*.jpg')))}.jpg"
        save_path = self.images_dir / filename
        
        # Save image
        cv2.imwrite(str(save_path), image)
        logger.info(f"Saved image to {save_path}")
        
        return str(save_path)
        
    def create_annotation(self, image_path: str, pore_locations: List[Tuple[int, int]]) -> str:
        """
        Create and save annotation for an image.
        
        Args:
            image_path: Path to the image
            pore_locations: List of (x, y) coordinates of pores
            
        Returns:
            Path to the saved annotation
        """
        # Create annotation dictionary
        annotation = {
            'image_path': str(image_path),
            'pore_locations': pore_locations
        }
        
        # Generate annotation filename
        filename = Path(image_path).stem + '.json'
        save_path = self.annotations_dir / filename
        
        # Save annotation
        with open(save_path, 'w') as f:
            json.dump(annotation, f)
            
        logger.info(f"Saved annotation to {save_path}")
        return str(save_path)
        
    def prepare_dataset(self, train_ratio: float = 0.8):
        """
        Prepare the dataset by splitting into train and validation sets.
        
        Args:
            train_ratio: Ratio of training data to total data
        """
        # Get all image paths
        image_paths = list(self.images_dir.glob('*.jpg'))
        np.random.shuffle(image_paths)
        
        # Split into train and validation
        split_idx = int(len(image_paths) * train_ratio)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        
        # Create train and validation directories
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Move files to appropriate directories
        for path in train_paths:
            # Move image
            new_path = train_dir / path.name
            path.rename(new_path)
            
            # Move corresponding annotation
            ann_path = self.annotations_dir / (path.stem + '.json')
            if ann_path.exists():
                ann_path.rename(train_dir / ann_path.name)
                
        for path in val_paths:
            # Move image
            new_path = val_dir / path.name
            path.rename(new_path)
            
            # Move corresponding annotation
            ann_path = self.annotations_dir / (path.stem + '.json')
            if ann_path.exists():
                ann_path.rename(val_dir / ann_path.name)
                
        logger.info(f"Dataset prepared with {len(train_paths)} training and {len(val_paths)} validation images")

def main():
    # Initialize data collector
    collector = DataCollector('data/pore_dataset')
    
    # Example usage:
    # 1. Collect images
    # image_path = collector.collect_image('path/to/image.jpg', 'cheek')
    
    # 2. Create annotations
    # pore_locations = [(100, 100), (200, 200)]  # Example pore locations
    # collector.create_annotation(image_path, pore_locations)
    
    # 3. Prepare dataset
    # collector.prepare_dataset()
    
    logger.info("Data collection script ready!")

if __name__ == '__main__':
    main() 