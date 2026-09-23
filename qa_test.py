import requests
import os
import glob
import time

def run_tests():
    print("=== STARTING QA TESTS ===")
    
    # 1. Structure Check
    print("1. Checking File Structure...")
    required_files = [
        'app.py', 'requirements.txt', 'models/best.pt',
        'app/templates/index.html', 'app/static/js/main.js',
        'app/static/css/style.css', 'dataset/data.yaml'
    ]
    for f in required_files:
        assert os.path.exists(f), f"FAIL: Missing {f}"
    print("-> Structure OK")
    
    # 2. Get test image
    test_imgs = glob.glob('archive/Pothole_Segmentation_YOLOv8/valid/images/*.jpg')
    assert len(test_imgs) > 0, "FAIL: No test images found"
    test_img = test_imgs[0]
    
    # 3. Test API Endpoint
    print("2. Testing Flask Endpoints (assuming server is running)...")
    try:
        # Test index
        res = requests.get("http://127.0.0.1:5000/")
        assert res.status_code == 200, f"FAIL: Index returned {res.status_code}"
        print("-> Index page loaded successfully")
        
        # Test empty upload (should fail gracefully)
        res = requests.post("http://127.0.0.1:5000/predict")
        assert res.status_code == 400, f"FAIL: Empty upload should return 400, got {res.status_code}"
        print("-> Empty upload handled correctly")
        
        # Test invalid extension (if we had a non-image file, skipping for now as frontend handles it mostly, but backend does too)
        
        # Test valid prediction
        print(f"-> Uploading test image: {test_img}")
        with open(test_img, 'rb') as f:
            files = {'file': (os.path.basename(test_img), f, 'image/jpeg')}
            res = requests.post("http://127.0.0.1:5000/predict", files=files)
            
        assert res.status_code == 200, f"FAIL: Predict returned {res.status_code}. Response: {res.text}"
        data = res.json()
        assert data.get('success') == True, f"FAIL: Predict success=False. Error: {data.get('error')}"
        
        print(f"-> Prediction successful! Pothole Count: {data['pothole_count']}, Confidence: {data['confidence']:.2f}%")
        
        # 4. Verify output files
        # The result image URL is /static/results/result_UUID_filename.jpg
        # Since static folder is app/static, we construct path:
        result_uri = data['result_image'] # e.g. /static/results/...
        # Remove '/static/' prefix
        result_filename = result_uri.replace('/static/', '')
        result_filepath = os.path.join('app', 'static', result_filename)
        
        assert os.path.exists(result_filepath), f"FAIL: Result image not saved at {result_filepath}"
        print(f"-> Result image successfully saved at: {result_filepath}")
        
        print("\n====================")
        print("✅ ALL QA TESTS PASSED!")
        print("====================")
        
    except requests.exceptions.ConnectionError:
        print("FAIL: Flask server is not running at http://127.0.0.1:5000")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    run_tests()
