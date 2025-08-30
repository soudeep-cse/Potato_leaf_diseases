from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    status: str
    disease: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
