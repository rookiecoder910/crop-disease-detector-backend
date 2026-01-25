from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Initialize FastAPI app
app = FastAPI(title="Crop Disease Detection API")

# Add CORS middleware to allow requests from browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load your trained model (Docker-safe absolute path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "Model.hdf5")

print(" ** Loading model **")
print("Model path:", MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)
print(" ** Model loaded successfully **")


# PlantVillage dataset classes (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

IMG_HEIGHT, IMG_WIDTH = 224, 224  # Keep consistent with training

# Directory to save uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def preprocess_image(img_bytes: bytes):
    """Convert uploaded file bytes into preprocessed tensor."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.get("/")
async def root():
    return {"message": "Crop Disease Detection API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file bytes
        contents = await file.read()

        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # Preprocess & predict
        input_data = preprocess_image(contents)
        preds = model.predict(input_data)
        pred_index = np.argmax(preds)
        
        # Get the full class name for prediction
        prediction = CLASS_NAMES[pred_index]
        confidence = float(np.max(preds))
        
        # Split for additional info
        class_parts = prediction.split('___')
        crop = class_parts[0] if len(class_parts) > 0 else "Unknown"
        disease = class_parts[1].replace("_", " ").title() if len(class_parts) > 1 else "Unknown"

        result = {
            "prediction": prediction,  # Full class name that frontend expects
            "crop": crop,
            "disease": disease,
            "confidence": confidence
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
