import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import Dict
import tempfile
import shutil

from ..face_detection.detector import FaceDetector
from ..pore_detection.detector import PoreDetector
from ..llm_integration.analyzer import LLMAnalyzer

app = FastAPI(title="Smart Beauty Pore Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
face_detector = FaceDetector()
pore_detector = PoreDetector()
llm_analyzer = LLMAnalyzer()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)) -> Dict:
    """
    Analyze a selfie image for facial pores and generate recommendations.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Dictionary containing analysis results and recommendations
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Read image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect and align face
        aligned_face, landmarks = face_detector.detect_face(image)
        
        # Segment facial regions
        regions = face_detector.segment_facial_regions(aligned_face, landmarks)
        
        # Analyze each region
        region_analyses = []
        for region_name, region_image in regions.items():
            # Save region image temporarily
            region_path = f"{temp_path}_{region_name}.jpg"
            cv2.imwrite(region_path, region_image)
            
            # Get LLM analysis
            analysis = llm_analyzer.analyze_region(region_path, region_name)
            region_analyses.append(analysis)
            
            # Detect pores in region
            pore_data = pore_detector.detect_pores(region_image)
            
            # Generate mask
            mask = pore_detector.generate_mask(region_image, pore_data)
            
            # Save mask
            mask_path = f"{temp_path}_{region_name}_mask.jpg"
            cv2.imwrite(mask_path, mask)
            
            # Clean up temporary files
            os.remove(region_path)
            os.remove(mask_path)
        
        # Generate overall summary
        summary = llm_analyzer.generate_summary(region_analyses)
        
        # Clean up main temporary file
        os.remove(temp_path)
        
        return {
            "region_analyses": region_analyses,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 