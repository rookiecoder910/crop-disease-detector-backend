import requests
import os

def test_crop_disease_api():
    """
    Test the crop disease detection API
    """
    url = "http://127.0.0.1:8000/predict"
    
    # Test with a sample image (you can replace this path with your own image)
    # For now, let's test if the server is responding
    
    try:
        # Test the root endpoint first
        root_response = requests.get("http://127.0.0.1:8000/")
        print(f"✅ Root endpoint status: {root_response.status_code}")
        print(f"✅ Root response: {root_response.json()}")
        
        # Test the health endpoint if it exists
        try:
            health_response = requests.get("http://127.0.0.1:8000/health")
            print(f"✅ Health endpoint status: {health_response.status_code}")
        except:
            print("ℹ️  Health endpoint not available (that's ok)")
        
        print("\n📝 To test image prediction:")
        print("1. Save an image of a plant/crop in the 'uploads' folder")
        print("2. Update the 'image_path' variable below with your image filename")
        print("3. Run this script again")
        
        # Example image test (uncomment and modify when you have an image)
        # image_path = "uploads/sample_plant.jpg"  # Replace with your image
        # if os.path.exists(image_path):
        #     with open(image_path, "rb") as f:
        #         files = {"file": ("image.jpg", f, "image/jpeg")}
        #         response = requests.post(url, files=files)
        #         print(f"✅ Prediction status: {response.status_code}")
        #         print(f"🔍 Prediction result: {response.json()}")
        # else:
        #     print(f"❌ Image not found at: {image_path}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running on http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_crop_disease_api()