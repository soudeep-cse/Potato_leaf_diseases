from fastapi import APIRouter, UploadFile, File
from app.services.disease_detection import DiseaseDetectionService
from app.schemas.prediction import PredictionResponse
import yaml
import os

router = APIRouter()

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize service
disease_service = DiseaseDetectionService(
    model_path=config["model"]["path"],
    input_size=tuple(config["model"]["input_size"]),
    class_names=config["model"]["class_names"]
)

@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict disease in potato plant image
    """
    try:
        contents = await file.read()
        result = disease_service.predict(contents)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
