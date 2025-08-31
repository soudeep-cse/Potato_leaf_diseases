from pydantic import BaseModel
from typing import Optional, Dict

class PredictionResponse(BaseModel):
    status: str
    disease: Optional[str] = None
    confidence: Optional[float] = None
    all_confidences: Optional[Dict[str, float]] = None
    message: Optional[str] = None
