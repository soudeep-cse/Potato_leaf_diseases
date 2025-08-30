# Potato Plant Disease Detection API

This project implements a FastAPI-based REST API for detecting diseases in potato plants using deep learning. The system can identify three different conditions in potato plants: Early Blight, Late Blight, and Healthy plants.

## Features

- Fast and efficient disease detection
- RESTful API implementation using FastAPI
- Deep learning-based image classification
- Confidence score for predictions
- Easy-to-use API endpoint for image upload
- CORS support for cross-origin requests

## Technology Stack

- **Python 3.9+**
- **FastAPI** - Modern, fast web framework for building APIs
- **TensorFlow** - Deep learning framework for the classification model
- **Pillow** - Image processing library
- **Uvicorn** - Lightning-fast ASGI server
- **Pydantic** - Data validation using Python type annotations
- **PyYAML** - YAML file parsing for configuration

## Project Structure

```
app/
├── api/
│   └── v1/
│       └── endpoints/
│           ├── __init__.py
│           └── disease.py
├── model/
│   └── potato_disease_model.h5
├── schemas/
│   ├── __init__.py
│   └── prediction.py
├── services/
│   ├── __init__.py
│   └── disease_detection.py
├── config/
│   └── config.yaml
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/soudeep-cse/Potato_leaf_diseases.git
cd potato-disease-detection
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your trained model:

- Put your trained model file (`potato_disease_model.h5`) in the `app/model/` directory

## Configuration

The application can be configured through `config/config.yaml`. Key configurations include:

- Server host and port
- Model path
- Input image size
- Class names for disease detection

## Running the Application

1. Start the FastAPI server:

```bash
cd app
python main.py
```

Or alternatively:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. The API will be available at `http://localhost:8000`
3. Access the interactive API documentation at `http://localhost:8000/docs`

## API Usage

### Predict Disease Endpoint

```http
POST /api/v1/predict
```

- Request: Form data with an image file
- Response: JSON with disease prediction and confidence score

Example response:

```json
{
  "status": "success",
  "disease": "Early Blight",
  "confidence": 0.95
}
```

## Error Handling

The API includes comprehensive error handling for:

- Invalid file formats
- Processing errors
- Model prediction errors

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the FastAPI community for the excellent framework
- TensorFlow team for the deep learning tools
- Contributors and maintainers of all dependent packages
