import os
import shutil
import torch
import pandas as pd
from ultralytics import YOLO

def main():
    print("=== CHUẨN BỊ HUẤN LUYỆN LẠI (V2) ===")
    
    model_path = "models/best.pt"
    backup_path = "models/best_backup.pt"
    
    if os.path.exists(model_path) and not os.path.exists(backup_path):
        shutil.copy(model_path, backup_path)
        print(f"-> Đã backup model hiện tại sang: {backup_path}")

    print("\n=== KIỂM TRA PHẦN CỨNG ===")
    if torch.cuda.is_available():
        device = '0'
        print(f"-> Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("-> CẢNH BÁO: Không có GPU. Sử dụng CPU.")
        
    epochs = 50
    patience = 15
        
    print("\n=== BẮT ĐẦU HUẤN LUYỆN ===")
    last_checkpoint = "runs/segment/pothole_seg_v2/weights/last.pt"
    if os.path.exists(last_checkpoint):
        print(f"-> Tiếp tục huấn luyện từ checkpoint: {last_checkpoint}")
        model = YOLO(last_checkpoint)
        resume_flag = True
    else:
        print("-> Bắt đầu huấn luyện mới từ pre-trained model")
        model = YOLO('yolov8n-seg.pt')
        resume_flag = False
    
    results = model.train(
        data="dataset/data.yaml",
        epochs=epochs,
        patience=patience,
        imgsz=640,
        batch=4,
        device=device,
        project="runs/segment",
        name="pothole_seg_v2",
        exist_ok=True,
        resume=resume_flag,
        workers=0
    )
    
    print("\n=== ĐÁNH GIÁ MÔ HÌNH SAU HUẤN LUYỆN ===")
    best_weights = "runs/segment/pothole_seg_v2/weights/best.pt"
    if os.path.exists(best_weights):
        shutil.copy(best_weights, model_path)
        print(f"-> Đã ghi đè model mới nhất vào: {model_path}")
    else:
        print("-> Lỗi: Không tìm thấy file best.pt của đợt train này.")
        
    try:
        csv_path = "runs/segment/pothole_seg_v2/results.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            last_epoch = df.iloc[-1]
            print("\n=== THỐNG KÊ METRICS (EPOCH CUỐI) ===")
            print(f"Box Loss (Train): {last_epoch.get('train/box_loss', 'N/A'):.4f}")
            print(f"Seg Loss (Train): {last_epoch.get('train/seg_loss', 'N/A'):.4f}")
            print(f"Precision (Mask): {last_epoch.get('metrics/precision(M)', 'N/A'):.4f}")
            print(f"Recall (Mask): {last_epoch.get('metrics/recall(M)', 'N/A'):.4f}")
            print(f"mAP50 (Mask): {last_epoch.get('metrics/mAP50(M)', 'N/A'):.4f}")
            print(f"mAP50-95 (Mask): {last_epoch.get('metrics/mAP50-95(M)', 'N/A'):.4f}")
    except Exception as e:
        print(f"Lỗi khi đọc file metrics: {e}")

if __name__ == "__main__":
    main()
