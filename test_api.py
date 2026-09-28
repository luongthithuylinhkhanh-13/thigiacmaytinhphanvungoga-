import requests
import json
import io

url = "http://127.0.0.1:5000/detect-image"
files = {'file': ('test.jpg', b'dummy_image_content', 'image/jpeg')}
files = {'file': open('ảnh test.webp', 'rb')}

# We need to simulate login first! 
# Let's use a session.
session = requests.Session()

# 1. Register a test user
register_data = {'full_name': 'Test', 'email': 'test@example.com', 'password': 'password123', 'confirm_password': 'password123'}
session.post("http://127.0.0.1:5000/register", data=register_data)

# 2. Login
login_data = {'email': 'test@example.com', 'password': 'password123'}
session.post("http://127.0.0.1:5000/login", data=login_data)

# 3. Detect image
res = session.post(url, files=files, data={'show_mask': 'true', 'show_contour': 'true', 'show_conf': 'true', 'conf_threshold': '0.5'})
try:
    print(json.dumps(res.json(), indent=2))
except Exception as e:
    print("Failed to decode JSON:", res.text)
