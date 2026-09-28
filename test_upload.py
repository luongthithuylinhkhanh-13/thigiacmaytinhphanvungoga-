import requests
from io import BytesIO
from PIL import Image

def test_upload():
    s = requests.Session()
    # 1. We might not even need login if we can register a user first
    s.post("http://127.0.0.1:5000/register", data={"full_name": "Test", "email": "test@test.com", "password": "password", "confirm_password": "password"})
    s.post("http://127.0.0.1:5000/login", data={"email": "test@test.com", "password": "password"})
    
    # 2. Create webp file
    img = Image.new('RGB', (800, 600), color='blue')
    buf = BytesIO()
    img.save(buf, format='WEBP')
    buf.seek(0)
    
    # 3. Post to detect-image
    files = {'file': ('ảnh test.webp', buf, 'image/webp')}
    try:
        r = s.post("http://127.0.0.1:5000/detect-image", files=files, data={'show_mask': 'true'})
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
    except Exception as e:
        print("EXCEPTION:", str(e))

if __name__ == '__main__':
    test_upload()
