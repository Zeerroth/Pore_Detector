import os
import json
from typing import Dict, List
import openai
from dotenv import load_dotenv

load_dotenv()

class LLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key

    def analyze_region(self, image_path: str, region_name: str) -> Dict:
        """
        Analyze a facial region using GPT-4 Vision.
        
        Args:
            image_path: Path to the cropped region image
            region_name: Name of the facial region
            
        Returns:
            Dictionary containing analysis results
        """
        # Read image as base64
        with open(image_path, "rb") as image_file:
            base64_image = image_file.read()

        # Prepare the prompt
        prompt = f"""Analyze this {region_name} image for visible skin pores.
        Rate the severity of visible pores on a scale of 0-100.
        Describe the location and pattern of visible pores.
        Format your response as a JSON object with these fields:
        - severity: number (0-100)
        - description: string
        - recommendations: array of strings"""

        # Call GPT-4 Vision API
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Parse and return the response
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {
                "region": region_name,
                "severity": analysis["severity"],
                "description": analysis["description"],
                "recommendations": analysis["recommendations"]
            }
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

    def generate_summary(self, region_analyses: List[Dict]) -> Dict:
        """
        Generate a summary of all region analyses.
        
        Args:
            region_analyses: List of analysis results for each region
            
        Returns:
            Dictionary containing overall summary and recommendations
        """
        # Prepare the prompt
        prompt = f"""Based on these facial region analyses, provide a comprehensive summary:
        {json.dumps(region_analyses, indent=2)}
        
        Format your response as a JSON object with these fields:
        - overall_severity: number (0-100)
        - summary: string
        - key_concerns: array of strings
        - skincare_routine: array of strings"""

        # Call GPT-4 API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500
        )

        # Parse and return the response
        try:
            summary = json.loads(response.choices[0].message.content)
            return summary
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON") 