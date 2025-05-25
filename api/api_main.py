from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model(r"..\models\poteto_diseases.h5")
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Enable CORS for handling requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can be restricted to specific origins if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

def read_file_as_image(data) -> np.ndarray:
    """
    Reads an image file and returns it as a NumPy array after resizing it to the expected input shape.
    """
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # Resize to 256x256 as expected by the model
    # Uncomment the following lines if model needs RGB conversion and normalization:
    # image = image.convert("RGB")  # Convert to RGB format if not already
    # image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handles prediction request for an uploaded image.
    This function receives the image file, processes it, and returns the predicted class and confidence.
    """
    image = read_file_as_image(await file.read())  # Preprocess the uploaded image
    image_batch = np.expand_dims(image, axis=0)    # Add batch dimension for prediction

    # Get model predictions
    predictions = model.predict(image_batch)

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]  # Class with the highest confidence
    confidence = np.max(predictions[0])            # Confidence score

    # Return the prediction and confidence as JSON
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    # Start the FastAPI application
    uvicorn.run(app, host="localhost", port=8000)
