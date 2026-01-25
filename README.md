# 🌱 Crop Disease Detector – Backend (FastAPI + CNN)

This repository contains the **backend service** for the Crop Disease Detector application.  
It exposes a REST API built with **FastAPI** that uses a **trained CNN model** to predict crop diseases from uploaded images.

The backend is designed to be consumed by:
- 📱 Android (Jetpack Compose) mobile app
- 🌐 Any frontend that supports multipart image upload

---

## 🚀 Deployed API

🔗 **Base URL:**  
https://crop-disease-detector-backend-a16n.onrender.com/

🔗 **Prediction Endpoint:**  
POST /predict


> ⚠️ Replace the URL above with your actual deployed link.

---

## 🧠 Model Details

- Framework: **TensorFlow / Keras**
- Dataset: **PlantVillage**
- Input size: **224 × 224 RGB**
- Output: **38 crop–disease classes**
- Model format: `.hdf5`

---

## 📦 Tech Stack

- **FastAPI** – API framework
- **Uvicorn** – ASGI server
- **TensorFlow (CPU)** – Model inference
- **Pillow** – Image preprocessing
- **NumPy** – Numerical operations
- **Python Multipart** – File uploads
- **CORS Middleware** – Mobile/Web access

---

## 📡 API Usage

### 🔮 Predict Crop Disease

**Endpoint**
```http
POST /predict
Request

Content-Type: multipart/form-data

Field name: file

Value: image file (.jpg, .jpeg, .png)

Response

{
  "prediction": "Tomato___Late_blight",
  "crop": "Tomato",
  "disease": "Late Blight",
  "confidence": 0.92
}
🧪 Local Setup
1️⃣ Clone Repository
git clone https://github.com/rookiecoder910/crop-disease-detector-backend.git
cd crop-disease-detector-backend
2️⃣ Create Virtual Environment
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Server
uvicorn main:app --host 0.0.0.0 --port 8000
API will be available at:

http://127.0.0.1:8000
🛠️ Major Issues Faced & Fixes Applied
❌ Issue 1: API not responding from Android app
Cause: Backend service was powered off
Fix:

Added API health check

Verified server availability before debugging client

❌ Issue 2: Model file not found in deployment
Cause: Model file ignored by .gitignore and GitHub size limits
Fix:

Removed model from .gitignore

Used proper deployment strategy for large ML files

Ensured correct absolute model path (/app/model/Model.hdf5)

❌ Issue 3: Deployment failed due to memory limits
Cause: TensorFlow model exceeding free-tier memory
Fix:

Switched to tensorflow-cpu

Optimized model loading

Used Docker-based deployment

❌ Issue 4: CORS errors from frontend
Cause: Missing CORS configuration
Fix:

Added CORSMiddleware with allowed origins

📁 Project Structure
.
├── main.py               # FastAPI application
├── model/
│   └── Model.hdf5        # Trained CNN model
├── uploads/              # Uploaded images (runtime)
├── requirements.txt
├── Dockerfile
└── README.md
📌 Future Improvements
Add /health endpoint for monitoring

Add batch prediction support

Model optimization (quantization)

Logging & request tracing

Authentication for production use

👨‍💻 Author
Manas Kumar
🔗 GitHub: https://github.com/rookiecoder910
