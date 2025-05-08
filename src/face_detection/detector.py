import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def detect_face(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect face and landmarks in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (aligned_face, landmarks)
        """
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face mesh landmarks
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image")
            
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
            
        # Align face
        aligned_face = self._align_face(image, landmarks)
        
        return aligned_face, landmarks

    def segment_facial_regions(self, image: np.ndarray, landmarks: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Segment facial regions based on landmarks.
        
        Args:
            image: Input image
            landmarks: List of facial landmarks
            
        Returns:
            Dictionary of segmented regions
        """
        height, width = image.shape[:2]
        
        # Define region boundaries based on landmarks
        regions = {
            'left_cheek': self._get_cheek_region(landmarks, 'left', width, height),
            'right_cheek': self._get_cheek_region(landmarks, 'right', width, height),
            'nose': self._get_nose_region(landmarks, width, height),
            'forehead': self._get_forehead_region(landmarks, width, height),
            'chin': self._get_chin_region(landmarks, width, height)
        }
        
        # Crop regions
        cropped_regions = {}
        for region_name, (x1, y1, x2, y2) in regions.items():
            cropped_regions[region_name] = image[y1:y2, x1:x2]
            
        return cropped_regions

    def _align_face(self, image: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """Align face based on eye positions."""
        # Get eye landmarks
        left_eye = np.mean([(landmarks[33]['x'], landmarks[33]['y']),
                           (landmarks[133]['x'], landmarks[133]['y'])], axis=0)
        right_eye = np.mean([(landmarks[362]['x'], landmarks[362]['y']),
                            (landmarks[263]['x'], landmarks[263]['y'])], axis=0)
        
        # Calculate angle
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                    right_eye[0] - left_eye[0]))
        
        # Rotate image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_face = cv2.warpAffine(image, rotation_matrix, (width, height),
                                     flags=cv2.INTER_CUBIC)
        
        return aligned_face

    def _get_cheek_region(self, landmarks: List[Dict], side: str, width: int, height: int) -> Tuple[int, int, int, int]:
        """Get cheek region coordinates."""
        if side == 'left':
            points = [landmarks[123], landmarks[50], landmarks[36]]
        else:
            points = [landmarks[352], landmarks[280], landmarks[266]]
            
        x_coords = [int(p['x'] * width) for p in points]
        y_coords = [int(p['y'] * height) for p in points]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def _get_nose_region(self, landmarks: List[Dict], width: int, height: int) -> Tuple[int, int, int, int]:
        """Get nose region coordinates."""
        nose_points = [landmarks[1], landmarks[2], landmarks[3], landmarks[4]]
        x_coords = [int(p['x'] * width) for p in nose_points]
        y_coords = [int(p['y'] * height) for p in nose_points]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def _get_forehead_region(self, landmarks: List[Dict], width: int, height: int) -> Tuple[int, int, int, int]:
        """Get forehead region coordinates."""
        forehead_points = [landmarks[10], landmarks[67], landmarks[69]]
        x_coords = [int(p['x'] * width) for p in forehead_points]
        y_coords = [int(p['y'] * height) for p in forehead_points]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    def _get_chin_region(self, landmarks: List[Dict], width: int, height: int) -> Tuple[int, int, int, int]:
        """Get chin region coordinates."""
        chin_points = [landmarks[152], landmarks[175], landmarks[199]]
        x_coords = [int(p['x'] * width) for p in chin_points]
        y_coords = [int(p['y'] * height) for p in chin_points]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords)) 