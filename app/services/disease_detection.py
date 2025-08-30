import tensorflow as tf
import numpy as np
from PIL import Image
import io

class DiseaseDetectionService:
    def __init__(self, model_path: str, input_size: tuple, class_names: list):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = input_size
        self.class_names = class_names

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess the input image for model prediction"""
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize(self.input_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)

    def predict(self, image_bytes: bytes) -> dict:
        """Predict the disease in the potato plant image"""
        try:
            processed_image = self.preprocess_image(image_bytes)
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            return {
                "disease": self.class_names[predicted_class_idx],
                "confidence": confidence,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
