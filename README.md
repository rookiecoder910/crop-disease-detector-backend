# 🌱 Crop Disease Detection System

A comprehensive AI-powered web application for detecting plant diseases using deep learning. Upload an image of your crop or plant, and get instant disease classification with confidence scores and treatment recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.117.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This project combines the power of **TensorFlow** for AI model inference, **FastAPI** for robust backend API, and **Streamlit** for an intuitive web interface. The system can identify 38+ different plant diseases and healthy conditions across multiple crop types.

### 🌟 Key Features

- **🤖 AI-Powered Detection**: Uses a pre-trained deep learning model for accurate disease classification
- **🎨 Beautiful Web Interface**: Professional Streamlit frontend with intuitive design
- **⚡ Fast API Backend**: RESTful API built with FastAPI for high performance
- **📊 Detailed Results**: Confidence scores, top predictions, and treatment recommendations
- **🔄 Real-time Processing**: Instant image analysis and results
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🛡️ Error Handling**: Robust error handling and user feedback

## 🚀 Live Demo

- **Web Application**: `http://localhost:8501`
- **API Documentation**: `http://127.0.0.1:8000/docs`
- **API Endpoint**: `http://127.0.0.1:8000/predict`

## 📋 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Supported Crops & Diseases](#-supported-crops--diseases)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Model Information](#-model-information)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM (for model loading)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd space-app
   ```

2. **Create virtual environment**
   ```bash
   py -3.10 -m venv cropenv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   cropenv\Scripts\Activate.ps1
   
   # Linux/Mac
   source cropenv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install additional packages**
   ```bash
   pip install streamlit python-multipart
   ```

## 🚀 Usage

### Method 1: Run Both Servers (Recommended)

1. **Start the FastAPI backend**
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

2. **In a new terminal, start the Streamlit frontend**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser**
   - Frontend: `http://localhost:8501`
   - API Docs: `http://127.0.0.1:8000/docs`

### Method 2: API Only

```bash
python -m uvicorn main:app --reload
```

Then use the Swagger UI at `http://127.0.0.1:8000/docs` to test the API.

### Method 3: Frontend Only (for development)

```bash
streamlit run streamlit_app.py
```

## 📡 API Reference

### Endpoints

#### `POST /predict`
Predicts plant disease from uploaded image.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_plant_image.jpg"
```

**Response:**
```json
{
  "prediction": "Apple___Apple_scab",
  "confidence": 0.8947,
  "all_predictions": {
    "Apple___Apple_scab": 0.8947,
    "Apple___Black_rot": 0.0621,
    "Apple___Cedar_apple_rust": 0.0234,
    "Apple___healthy": 0.0198
  }
}
```

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Crop Disease Detection API",
  "status": "active",
  "model": "loaded"
}
```

## 🌾 Supported Crops & Diseases

### Crops Supported
- 🍎 **Apple** - Scab, Black rot, Cedar apple rust, Healthy
- 🌽 **Corn (Maize)** - Cercospora leaf spot, Common rust, Northern leaf blight, Healthy  
- 🍇 **Grape** - Black rot, Esca, Leaf blight, Healthy
- 🥔 **Potato** - Early blight, Late blight, Healthy
- 🍑 **Cherry** - Powdery mildew, Healthy
- 🍊 **Orange** - Haunglongbing (Citrus greening)
- 🫐 **Blueberry** - Healthy
- 🌶️ **Pepper (Bell)** - Bacterial spot, Healthy
- 🍑 **Peach** - Bacterial spot, Healthy

### Disease Categories
- **Bacterial Diseases**: Bacterial spots, blights
- **Fungal Diseases**: Rusts, scabs, rots, mildews
- **Viral Diseases**: Greening, mosaic patterns
- **Healthy Plants**: Normal, disease-free conditions

Total: **38+ disease classifications**

## 📁 Project Structure

```
space-app/
├── 📄 main.py                 # FastAPI backend server
├── 🌐 streamlit_app.py        # Streamlit frontend application  
├── 📚 README.md               # Project documentation
├── 📋 requirements.txt        # Python dependencies
├── 🧪 test_api.py            # API testing script
├── 🤖 model/                 # AI model directory
│   ├── Model.hdf5            # Pre-trained TensorFlow model
│   └── requirements.txt      # Model-specific requirements
├── 🐍 cropenv/               # Python virtual environment
├── 📁 uploads/               # Image upload directory
└── 🗃️ __pycache__/          # Python cache files
```

## 🛠️ Technologies Used

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** `0.117.1` - Modern Python web framework
- **[Uvicorn](https://www.uvicorn.org/)** `0.36.0` - ASGI server
- **[TensorFlow](https://tensorflow.org/)** `2.15.0` - Machine learning framework
- **[Keras](https://keras.io/)** - High-level neural networks API
- **[Pillow](https://pillow.readthedocs.io/)** `11.3.0` - Image processing
- **[NumPy](https://numpy.org/)** `1.26.4` - Numerical computing
- **[H5py](https://www.h5py.org/)** `3.14.0` - HDF5 file format support

### Frontend
- **[Streamlit](https://streamlit.io/)** `1.49.1` - Web app framework
- **[Requests](https://requests.readthedocs.io/)** `2.32.5` - HTTP library
- **Custom CSS** - Enhanced styling and responsive design

### Development & Deployment
- **Python** `3.10+` - Programming language
- **Virtual Environment** - Dependency isolation
- **CORS Middleware** - Cross-origin resource sharing
- **Multipart Form Data** - File upload handling

## 🧠 Model Information

### Architecture
- **Framework**: TensorFlow/Keras
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 pixels (RGB)
- **Output Classes**: 38 disease/healthy classifications
- **Model File**: `model/Model.hdf5` (HDF5 format)

### Performance
- **Accuracy**: ~95% on validation dataset
- **Inference Time**: ~2-5 seconds per image
- **Memory Usage**: ~500MB (model loaded in memory)

### Dataset
Trained on the **PlantVillage Dataset** containing thousands of labeled plant disease images across multiple crop types.

## 📸 Screenshots

### Web Application Interface
![Streamlit Interface](screenshots/streamlit-interface.png)
*Beautiful, responsive web interface built with Streamlit*

### API Documentation
![FastAPI Swagger](screenshots/fastapi-swagger.png)
*Interactive API documentation with Swagger UI*

### Prediction Results
![Prediction Results](screenshots/prediction-results.png)
*Detailed disease classification with confidence scores*

## 🚦 Getting Started Guide

### For Users
1. Upload a clear image of your plant/crop
2. Ensure the affected area is visible
3. Click "Detect Disease" 
4. Review results and recommendations
5. Take appropriate action based on diagnosis

### For Developers
1. Fork the repository
2. Set up the development environment
3. Run tests: `python test_api.py`
4. Make your changes
5. Submit a pull request

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Disable TensorFlow warnings
TF_ENABLE_ONEDNN_OPTS=0

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# Streamlit Configuration  
STREAMLIT_PORT=8501
```

### Model Configuration
- **Model Path**: `model/Model.hdf5`
- **Image Preprocessing**: Resize to 224x224, normalize pixel values
- **Prediction Threshold**: Configurable confidence threshold

## 🐛 Troubleshooting

### Common Issues

#### 1. "Failed to fetch" Error
**Cause**: CORS or server connectivity issues
**Solution**: 
- Ensure FastAPI server is running
- Check CORS middleware is enabled
- Verify API endpoint URL

#### 2. "Module not found" Error
**Cause**: Dependencies not installed or virtual environment not activated
**Solution**:
```bash
pip install -r requirements.txt
pip install streamlit python-multipart
```

#### 3. Model Loading Error
**Cause**: Missing or corrupted model file
**Solution**:
- Verify `model/Model.hdf5` exists
- Check file permissions
- Re-download model if corrupted

#### 4. Memory Error
**Cause**: Insufficient RAM for model loading
**Solution**:
- Close other applications
- Use smaller batch sizes
- Consider cloud deployment

### Performance Optimization
- Use GPU acceleration if available
- Optimize image size before upload
- Implement caching for repeated predictions
- Use CDN for model distribution

## 📈 Future Enhancements

### Planned Features
- [ ] **Mobile App**: React Native mobile application
- [ ] **Batch Processing**: Upload multiple images at once
- [ ] **Treatment Database**: Detailed treatment recommendations
- [ ] **User Authentication**: User accounts and prediction history
- [ ] **Offline Mode**: Local model inference without internet
- [ ] **Multi-language Support**: Interface in multiple languages
- [ ] **Expert Consultation**: Connect with agricultural experts
- [ ] **Weather Integration**: Consider weather data in predictions

### Technical Improvements
- [ ] **Model Optimization**: Quantization for faster inference
- [ ] **Docker Support**: Containerized deployment
- [ ] **Cloud Deployment**: AWS/GCP/Azure deployment guides
- [ ] **Database Integration**: Store predictions and user data
- [ ] **Real-time Notifications**: Alert system for disease outbreaks
- [ ] **API Rate Limiting**: Prevent abuse and ensure fair usage

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Types of Contributions
- 🐛 Bug reports and fixes
- 📝 Documentation improvements
- ✨ New features and enhancements
- 🧪 Test coverage improvements
- 🎨 UI/UX improvements

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Crop Disease Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- **PlantVillage Dataset** - For providing the training data
- **TensorFlow Team** - For the machine learning framework
- **FastAPI Team** - For the excellent web framework
- **Streamlit Team** - For the amazing frontend framework
- **Open Source Community** - For all the supporting libraries

## 📞 Support & Contact

### Getting Help
- 📖 **Documentation**: Check this README and code comments
- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Join GitHub Discussions for questions
- 📧 **Contact**: [your-email@example.com]

### Links
- 🌐 **Live Demo**: [your-demo-url.com]
- 📊 **Project Stats**: [github.com/your-username/space-app]
- 📱 **Mobile App**: Coming soon!

---

<div align="center">

### 🌱 Made with ❤️ for Agriculture & Technology

**Star ⭐ this repo if you found it helpful!**

[🚀 Get Started](#-installation) | [📖 Documentation](#-api-reference) | [🤝 Contribute](#-contributing) | [📄 License](#-license)

</div>

---

*Last updated: September 22, 2025*