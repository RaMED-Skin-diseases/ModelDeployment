from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

# === Load TensorFlow Model ===
TF_MODEL_PATH = os.environ.get("MODEL_PATH", "model/final_model.h5")
try:
    tf_model = tf.keras.models.load_model(TF_MODEL_PATH)
    feature_extractor = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg")
except Exception as e:
    print(f"[ERROR] Failed to load TensorFlow model: {e}")
    tf_model = None

# === Class Names ===
TF_CLASSES = [
    'Eczema', 'Melanoma', 'Atopic Dermatitis',
    'Basal Cell Carcinoma', 'Melanocytic Nevi', 'Benign Keratosis'
]

# === Preprocessing Functions ===
def preprocess_for_tf(image: Image.Image):
    # Resize and preprocess the image
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    # Extract features using ResNet50
    features = feature_extractor.predict(image_array)
    return features

# === Prediction Endpoint ===
@app.post("/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        
        # Initialize response variables
        tf_class, tf_prob = None, 0

        # === TensorFlow Prediction ===
        if tf_model:
            tf_features = preprocess_for_tf(image)
            tf_preds = tf_model.predict(tf_features)[0] * 100
            tf_max_idx = np.argmax(tf_preds)
            tf_class = TF_CLASSES[tf_max_idx]
            tf_prob = tf_preds[tf_max_idx]

        # Prepare response
        response = {
            "prediction": {
                "model": "TensorFlow",
                "class": tf_class,
                "probability": f"{tf_prob:.2f}%",
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))