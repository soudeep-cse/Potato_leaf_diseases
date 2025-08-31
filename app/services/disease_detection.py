import tensorflow as tf
import numpy as np
from PIL import Image
import io
import datetime

class DiseaseDetectionService:
    def __init__(self, model_path: str, input_size: tuple, class_names: list):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = input_size
        self.class_names = class_names

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess the input image for model prediction"""
        # Open and convert image to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize image
        image = image.resize(self.input_size)
        
        # Convert to numpy array and normalize
        image = np.array(image) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image

    def predict(self, image_bytes: bytes) -> dict:
        """Predict the disease in the potato plant image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get confidence scores for all classes
            class_confidences = {
                class_name: float(conf) 
                for class_name, conf in zip(self.class_names, predictions[0])
            }
            
            # Determine plant health status and recommendation
            predicted_class = self.class_names[predicted_class_idx]
            recommendation = ""
            if predicted_class == "Healthy":
                recommendation = "The plant appears healthy. Continue regular maintenance."
            elif predicted_class == "Early Blight":
                recommendation = "Early Blight detected. Remove affected leaves and apply appropriate fungicide."
            else:  # Late Blight
                recommendation = "Late Blight detected. Immediate action required. Apply fungicide and isolate affected plants."
            
            return {
                "status": "success",
                "prediction": {
                    "disease": predicted_class,
                    "confidence": round(confidence * 100, 2),  # Convert to percentage
                    "health_status": "Healthy" if predicted_class == "Healthy" else "Diseased"
                },
                "detailed_analysis": {
                    "all_confidences": {
                        disease: round(conf * 100, 2)  # Convert to percentage
                        for disease, conf in class_confidences.items()
                    },
                    "recommendation": recommendation
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
