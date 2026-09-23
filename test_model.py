import os
import glob
from ultralytics import YOLO

def main():
    model_path = "models/best.pt"
    
    print("=== KIỂM TRA MÔ HÌNH ===")
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại {model_path}!")
        return
        
    print("Đang load mô hình...")
    try:
        model = YOLO(model_path)
        print("-> Load mô hình THÀNH CÔNG!")
        
        # Test thử trên 1 ảnh validation
        print("\n=== TEST INFERENCE ===")
        val_img_dir = "archive/Pothole_Segmentation_YOLOv8/valid/images"
        imgs = glob.glob(os.path.join(val_img_dir, "*.jpg"))
        
        if imgs:
            test_img = imgs[0]
            print(f"Thử nghiệm dự đoán trên ảnh: {test_img}")
            results = model.predict(source=test_img, save=False, verbose=False)
            
            # Kiểm tra xem có kết quả boxes/masks trả về không
            if len(results) > 0:
                print(f"-> Dự đoán thành công! Số lượng vật thể tìm thấy: {len(results[0])}")
            else:
                print("-> Dự đoán hoàn tất nhưng không tìm thấy đối tượng nào.")
        else:
            print("Không tìm thấy ảnh test trong thư mục validation.")
            
    except Exception as e:
        print(f"Lỗi trong quá trình load hoặc test mô hình: {e}")

if __name__ == "__main__":
    main()
