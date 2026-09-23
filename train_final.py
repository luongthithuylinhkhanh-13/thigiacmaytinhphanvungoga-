import os
import shutil
import torch
import pandas as pd
from ultralytics import YOLO

def main():
    print("=== CHUẨN BỊ HUẤN LUYỆN CHÍNH THỨC ===")
    
    model_path = "models/best.pt"
    backup_path = "models/best_backup2.pt"
    
    if os.path.exists(model_path):
        shutil.copy(model_path, backup_path)
        print(f"-> Đã backup model sang: {backup_path}")

    print("\n=== KIỂM TRA PHẦN CỨNG ===")
    if torch.cuda.is_available():
        device = '0'
        print(f"-> Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        batch_size = 16
    else:
        device = 'cpu'
        print("-> Không có GPU. Chạy CPU cấu hình tối ưu (imgsz=320).")
        batch_size = 8
        
    epochs = 30
    patience = 15
    imgsz = 320 # Dataset là 320x320
        
    print(f"\n=== BẮT ĐẦU HUẤN LUYỆN ({epochs} Epochs) ===")
    model = YOLO('yolov8n-seg.pt') # Bắt đầu từ đầu để tránh lỗi checkpoint cũ
    
    results = model.train(
        data="dataset/data.yaml",
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="runs/segment",
        name="pothole_seg_final",
        exist_ok=True,
        workers=0
    )
    
    print("\n=== ĐÁNH GIÁ MÔ HÌNH SAU HUẤN LUYỆN ===")
    best_weights = "runs/segment/pothole_seg_final/weights/best.pt"
    if os.path.exists(best_weights):
        shutil.copy(best_weights, model_path)
        print(f"-> Đã ghi đè model tốt nhất vào: {model_path}")
        
    try:
        csv_path = "runs/segment/pothole_seg_final/results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            last_epoch = df.iloc[-1]
            print("\n=== THỐNG KÊ METRICS TỔNG HỢP ===")
            print(f"Box Loss: {last_epoch.get('train/box_loss', 'N/A')}")
            print(f"Seg Loss: {last_epoch.get('train/seg_loss', 'N/A')}")
            print(f"Precision (Mask): {last_epoch.get('metrics/precision(M)', 'N/A')}")
            print(f"Recall (Mask): {last_epoch.get('metrics/recall(M)', 'N/A')}")
            print(f"mAP50 (Mask): {last_epoch.get('metrics/mAP50(M)', 'N/A')}")
            print(f"mAP50-95 (Mask): {last_epoch.get('metrics/mAP50-95(M)', 'N/A')}")
    except Exception as e:
        print(f"Lỗi đọc metrics: {e}")

if __name__ == "__main__":
    main()
