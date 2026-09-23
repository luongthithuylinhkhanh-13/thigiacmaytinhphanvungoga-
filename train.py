import os
import torch
from ultralytics import YOLO
import shutil

def main():
    print("=== KIỂM TRA PHẦN CỨNG ===")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        device = '0'
        epochs = 10  # Train kỹ hơn nếu có GPU
        imgsz = 640
        batch = 8
    else:
        print("LƯU Ý: Không tìm thấy GPU! Sẽ sử dụng CPU.")
        print("Cấu hình nhẹ (Fast Demo) được áp dụng để tránh treo máy.")
        device = 'cpu'
        epochs = 1   # Train 1 epoch để demo tạo ra file best.pt nhanh chóng
        imgsz = 320  # Resize ảnh nhỏ lại để train nhanh
        batch = 4

    # Khởi tạo mô hình YOLOv8 segmentation (bản nano cho nhẹ và nhanh)
    print("\n=== KHỞI TẠO MÔ HÌNH ===")
    model = YOLO("yolov8n-seg.pt")
    
    # Bắt đầu huấn luyện
    print("\n=== BẮT ĐẦU HUẤN LUYỆN ===")
    model.train(
        data="dataset/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name="pothole_seg",
        exist_ok=True, # Ghi đè thư mục runs cũ nếu có
        workers=0      # Tránh lỗi đa luồng trên một số máy Windows
    )
    
    # Lưu best model vào thư mục models/
    print("\n=== LƯU MÔ HÌNH ===")
    best_weights = "runs/pothole_seg/weights/best.pt"
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weights, "models/best.pt")
        print("-> Đã lưu mô hình tốt nhất vào: models/best.pt")
    else:
        print("-> Lỗi: Không tìm thấy file best.pt sau khi train.")

if __name__ == "__main__":
    main()
