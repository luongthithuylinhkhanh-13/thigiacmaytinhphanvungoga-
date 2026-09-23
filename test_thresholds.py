import requests
import os
import glob
import time

def test_case(test_num, conf, iou, test_img):
    print(f"\n--- TEST {test_num} ---")
    print(f"Gửi cấu hình: conf={conf}, iou={iou}")
    
    with open(test_img, 'rb') as f:
        files = {'file': (os.path.basename(test_img), f, 'image/jpeg')}
        data = {
            'conf_threshold': str(conf),
            'iou_threshold': str(iou)
        }
        res = requests.post("http://127.0.0.1:5000/predict", files=files, data=data)
        
    assert res.status_code == 200, f"Lỗi HTTP: {res.status_code}"
    res_data = res.json()
    assert res_data['success'] == True, "Predict thất bại"
    
    count = res_data['pothole_count']
    avg_conf = res_data['confidence']
    print(f"Kết quả -> Số lượng ổ gà: {count}, Average Confidence: {avg_conf:.2f}%")
    return count, avg_conf

def run_tests():
    test_imgs = glob.glob('archive/Pothole_Segmentation_YOLOv8/valid/images/*.jpg')
    if not test_imgs:
        print("Không tìm thấy ảnh test")
        return
    test_img = test_imgs[0]
    
    try:
        # TEST 1
        c1, a1 = test_case(1, 0.5, 0.3, test_img)
        # TEST 2
        c2, a2 = test_case(2, 0.7, 0.3, test_img)
        # TEST 3
        c3, a3 = test_case(3, 0.3, 0.5, test_img)
        
        print("\n=== TỔNG KẾT ===")
        print(f"Conf 0.5 => Đếm: {c1}")
        print(f"Conf 0.7 => Đếm: {c2}")
        print(f"Conf 0.3 => Đếm: {c3}")
        
        # Verify logic: higher threshold should yield fewer or equal bounding boxes
        assert c2 <= c1 <= c3, "Lỗi logic: Ngưỡng tự tin cao hơn nhưng lại ra nhiều object hơn!"
        print("-> TẤT CẢ TEST ĐỀU HOẠT ĐỘNG HOÀN HẢO!")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    run_tests()
