from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

tflite = tf.lite
print("Using TensorFlow Lite from TensorFlow")


app = FastAPI(title="Crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Disease detection model (TFLite)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.tflite")

print(" ** Loading TFLite disease model **")
print("Model path:", MODEL_PATH)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(" ** TFLite disease model loaded successfully **")

#Plant / Leaf detector model (Keras .h5) 
PLANT_DETECTOR_PATH = os.path.join(BASE_DIR, "model", "plant_detector.h5")

print(" ** Loading plant detector model **")
print("Plant detector path:", PLANT_DETECTOR_PATH)

plant_model = tf.keras.models.load_model(PLANT_DETECTOR_PATH)

# Auto-detect output shape to correctly interpret predictions
_output_shape = plant_model.output_shape  # e.g. (None, 1) or (None, 2)
_num_classes = _output_shape[-1]
print(f" ** Plant detector loaded — output shape: {_output_shape} | num_classes: {_num_classes} **")

print(" ** Plant detector model loaded successfully **")


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

IMG_HEIGHT = 224
IMG_WIDTH = 224


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes into a normalised (1, 224, 224, 3) float32 array."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def is_leaf_image(img_bytes: bytes) -> bool:
    """
    Run plant_detector.h5 to decide whether the image shows a crop/leaf.

    Handles two common training conventions automatically:
      • Binary (1 output neuron, sigmoid):
            output > 0.5  →  leaf   (assumes class-1 / positive class = leaf)
      • Two-class softmax (2 output neurons):
            argmax == 0   →  leaf   (assumes class-0 = leaf, class-1 = non-leaf)

    If your model was trained with the opposite convention flip the condition here.
    """
    img_array = preprocess_image(img_bytes)
    pred = plant_model.predict(img_array, verbose=0)

    if _num_classes == 1:
        # Binary sigmoid — single value in [0, 1]
        confidence = float(pred[0][0])
        return confidence > 0.5
    else:
        # Softmax — class-0 is assumed to be "leaf / plant"
        predicted_class = int(np.argmax(pred[0]))
        return predicted_class == 0


@app.get("/")
async def root():
    return {"message": "Crop Disease Detection API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # check if it's a leaf/crop image
        if not is_leaf_image(contents):
            return JSONResponse(content={
                "is_leaf": False,
                "message": "Please upload a leaf image for correct diagnosis"
            })

     
        input_data = preprocess_image(contents)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        pred_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        prediction = CLASS_NAMES[pred_index]
        crop, disease = prediction.split("___")
        disease = disease.replace("_", " ").title()

        return JSONResponse(content={
            "is_leaf": True,
            "prediction": prediction,
            "crop": crop,
            "disease": disease,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
