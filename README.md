# Smart Beauty Pore Detection

An intelligent system for detecting and analyzing facial pores using computer vision and LLM technology.

## Features

- Face detection and alignment
- Facial region segmentation (cheeks, nose, forehead, chin)
- Pore detection using multimodal LLM analysis
- Visual mask generation for detected pores
- Skincare recommendations based on analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

- `src/`
  - `face_detection/` - Face detection and alignment modules
  - `pore_detection/` - Pore detection and analysis
  - `llm_integration/` - LLM interface and analysis
  - `visualization/` - Mask generation and visualization
  - `api/` - FastAPI backend
- `models/` - Pre-trained models
- `tests/` - Unit tests
- `data/` - Sample data and test images

## Usage

1. Start the API server:
```bash
uvicorn src.api.main:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs`

## License

MIT License 